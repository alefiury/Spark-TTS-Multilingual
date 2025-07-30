import os
import re
import sys
import time
import itertools
from pathlib import Path
from typing import List, Dict, Any
from typing import Tuple, Optional, Generator

# Ensure the parent directory is in the path for SparkTTS import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM

from cli.SparkTTS import SparkTTS


class SparkTTSStreaming(SparkTTS):
    """
    Spark-TTS with streaming support for real-time text-to-speech generation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle_chunks(
        self,
        wav_chunk: np.ndarray,
        wav_overlap: Optional[np.ndarray],
        overlap_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle chunk formatting with crossfade in streaming mode

        Args:
            wav_chunk: Current audio chunk
            wav_overlap: Previous overlap section for crossfading
            overlap_len: Number of samples to overlap

        Returns:
            Tuple of (processed_chunk, new_overlap)
        """
        if wav_overlap is not None and len(wav_chunk) > overlap_len:
            # Create crossfade
            fade_out = np.linspace(1.0, 0.0, overlap_len)
            fade_in = np.linspace(0.0, 1.0, overlap_len)

            # Apply crossfade
            wav_chunk[:overlap_len] = (
                wav_overlap * fade_out +
                wav_chunk[:overlap_len] * fade_in
            )

        # Save overlap for next chunk
        new_overlap = wav_chunk[-overlap_len:] if len(wav_chunk) > overlap_len else wav_chunk

        return wav_chunk, new_overlap

    @torch.no_grad()
    def inference_stream(
        self,
        text: str,
        prompt_speech_path: Path = None,
        prompt_text: str = None,
        gender: str = None,
        pitch: str = None,
        speed: str = None,
        stream_chunk_size: int = 20,
        overlap_samples: float = 1024,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 3000,
        language: str = "en",
        use_multilingual: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        """
        Performs streaming inference to generate speech from text.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            stream_chunk_size (int): Number of tokens to accumulate before generating audio chunk
            overlap_samples (int): Duration of overlap between chunks in samples
            temperature (float): Sampling temperature for controlling randomness.
            top_k (int): Top-k sampling parameter.
            top_p (float): Top-p (nucleus) sampling parameter.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Yields:
            np.ndarray: Audio chunks as they are generated
        """
        # Prepare input prompt
        if gender is not None:
            prompt = self.process_prompt_control(
                gender,
                pitch,
                speed,
                text,
                language=language,
                use_multilingual=use_multilingual,
            )
            # For control mode, we'll extract global tokens from the generated text
            global_token_ids = None
        else:
            prompt, global_token_ids = self.process_prompt(
                text,
                prompt_speech_path,
                prompt_text,
                language=language,
                use_multilingual=use_multilingual
            )

        # Tokenize input
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        # Calculate overlap samples
        # overlap_samples = int(overlap_duration * self.sample_rate)

        # Initialize variables for streaming
        generated_tokens = []
        wav_overlap = None
        all_generated_ids = model_inputs.input_ids[0].tolist()

        # Use the model's generate method with streaming callback
        def process_tokens(input_ids, scores, **kwargs):
            """Callback function called after each token generation"""
            nonlocal generated_tokens, wav_overlap, all_generated_ids, global_token_ids

            # Get the last generated token
            last_token_id = input_ids[0, -1].item()
            all_generated_ids.append(last_token_id)

            # Decode to check if it's a semantic token
            token_str = self.tokenizer.decode([last_token_id])

            # Extract semantic token if present
            if "bicodec_semantic_" in token_str:
                match = re.search(r"bicodec_semantic_(\d+)", token_str)
                if match:
                    semantic_id = int(match.group(1))
                    generated_tokens.append(semantic_id)
                    print(f"DEBUG: Found semantic token {semantic_id}, total: {len(generated_tokens)}")

            # Extract global tokens for control mode if not yet extracted
            if gender is not None and global_token_ids is None:
                generated_text = self.tokenizer.decode(all_generated_ids)
                global_matches = re.findall(r"bicodec_global_(\d+)", generated_text)
                if global_matches:
                    global_token_ids = torch.tensor(
                        [[int(g) for g in global_matches]],
                        dtype=torch.long
                    ).unsqueeze(0).to(self.device)

            return False  # Continue generation

        # Generate tokens with callback
        # Since transformers doesn't support streaming callbacks directly, a custom generation loop is used
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask
        past_key_values = None

        for step in range(max_new_tokens):
            # Forward pass
            outputs = self.model(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, top_k)
                min_top_k = top_k_values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_top_k,
                    torch.tensor(-float('inf'), device=logits.device),
                    logits
                )

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, -float('inf'))

            # Sample from the filtered distribution
            probs = torch.softmax(logits, dim=-1)

            # Check for numerical issues
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                # print("Warning: Invalid probabilities detected, using argmax instead")
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            # Check for EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Update sequences
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=1)
            all_generated_ids.append(next_token.item())

            generated_tokens.append(next_token.item())

            # Generate audio chunk when we have enough tokens
            if len(generated_tokens) >= stream_chunk_size:
                predicts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                # Skip if we don't have global tokens yet
                if global_token_ids is not None:
                    pred_semantic_ids = (
                        torch.tensor([int(re.findall(r"bicodec_semantic_(\d+)", token)[0]) for token in  predicts])
                        .long()
                        .unsqueeze(0)
                    ).to(self.device)

                    # Generate audio for this chunk
                    wav_chunk = self.audio_tokenizer.detokenize(
                        global_token_ids.squeeze(0),
                        pred_semantic_ids[:, :stream_chunk_size]
                    )

                    # Handle overlap and crossfade
                    wav_chunk, wav_overlap = self.handle_chunks(
                        wav_chunk, wav_overlap, overlap_samples
                    )

                    # Remove processed tokens from buffer
                    generated_tokens = generated_tokens[stream_chunk_size:]

                    yield wav_chunk

        # Process any remaining tokens
        if generated_tokens and global_token_ids is not None:
            predicts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # Ensure we have only semantic tokens inside predicts
            predicts = [token for token in predicts if "bicodec_semantic_" in token]

            pred_semantic_ids = (
                torch.tensor([int(re.findall(r"bicodec_semantic_(\d+)", token)[0]) for token in  predicts])
                .long()
                .unsqueeze(0)
            ).to(self.device)

            wav_chunk = self.audio_tokenizer.detokenize(
                global_token_ids.squeeze(0),
                pred_semantic_ids
            )

            # Final chunk doesn't need overlap handling
            yield wav_chunk


    def measure_time_to_first_chunk(self, *args, **kwargs) -> Tuple[float, Generator[np.ndarray, None, None]]:
        """
        Run `inference_stream`, measure “time‑to‑first‑token” (chunk) (TTFT) in seconds,
        and return (ttf, generator).

        The returned generator starts with the first chunk that was already
        fetched for timing, so downstream code can iterate normally:

            ttf, stream = model.measure_time_to_first_chunk(...)
            print(f"TTF: {ttf:.2f}s")
            for i, chunk in enumerate(stream):
                ...

        If generation ends before any chunk is produced, TTF is 0 and the
        generator is empty.
        """
        gen = self.inference_stream(*args, **kwargs)
        warmup_chunk = next(gen)

        start = time.perf_counter()
        true_first_chunk = next(gen)
        ttf = time.perf_counter() - start

        full_stream = itertools.chain([true_first_chunk], gen)
        return ttf, full_stream


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparkTTSStreaming(
        model_dir=Path("/raid/aluno_alef/Spark-TTS-finetune/checkpoints-spark-tts-multilingual"),
        device=device
    )

    text = "Embora estivesse chovendo, eles decidiram passear na floresta."

    prompt_wav = Path("/raid/aluno_alef/Spark-TTS/ermis_11labs-0000-0003.wav")
    prompt_text = (
        "A união faz a força, já dizia minha avó. \
            E quando a gente se junta para resolver \
            os problemas do bairro, as coisas fluem melhor."
    )

    language = "pt"
    use_multilingual = True
    stream_chunk_size = 30
    overlap_samples = 640  # ~40 ms overlap

    num_runs = 100

    print("-"*100)

    # ttfs = []

    # for _ in tqdm(range(num_runs), total=num_runs, desc="Measuring TTF"):
    #     ttf, generator = model.measure_time_to_first_chunk(
    #         text=text,
    #         prompt_speech_path=prompt_wav,
    #         prompt_text=prompt_text,
    #         stream_chunk_size=stream_chunk_size,
    #         overlap_samples=overlap_samples,
    #         language=language,
    #         use_multilingual=use_multilingual,
    #     )
    #     ttfs.append(ttf)

    # ttfs = np.array(ttfs)
    # print(f"⏱️  Average time to first chunk: {ttfs.mean():.2f} s")
    # print(f"⏱️  Min time to first chunk: {ttfs.min():.2f} s")
    # print(f"⏱️  Max time to first chunk: {ttfs.max():.2f} s")
    # print(f"⏱️  Std time to first chunk: {ttfs.std():.2f} s")

    generator = model.inference_stream(
        text=text,
        prompt_speech_path=prompt_wav,
        prompt_text=prompt_text,
        stream_chunk_size=stream_chunk_size,     # emit every 30 semantic tokens
        overlap_samples=overlap_samples,      # 100 ms crossfade
        language="pt",
        use_multilingual=True,
    )

    os.makedirs("out", exist_ok=True)

    print("Generating voice‑cloned stream...")

    audio_chunks = []
    for i, chunk in enumerate(generator, start=0):
        sf.write(
            f"out/streaming_voice_cloned_output_{i:03d}.wav",
            chunk,
            model.sample_rate
        )
        print(f"  Chunk {i}: shape={chunk.shape}")
        audio_chunks.append(chunk)

    print(audio_chunks)

    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        sf.write(
            "out/streaming_output.wav",
            full_audio,
            model.sample_rate
        )
        duration = len(full_audio) / model.sample_rate
        print(f"→ Wrote streaming_output.wav ({duration:.2f} s)")


if __name__ == "__main__":
    main()