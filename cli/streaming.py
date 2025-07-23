import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import torch
import numpy as np
from typing import Tuple, Optional, Generator
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from sparktts.utils.file import load_config
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP

from cli.SparkTTS import SparkTTS


class SparkTTSStreaming:
    """
    Spark-TTS with streaming support for real-time text-to-speech generation.
    """

    def __init__(self, model_dir: Path, device: torch.device = torch.device("cuda:0")):
        """
        Initializes the SparkTTS model with streaming capabilities.

        Args:
            model_dir (Path): Directory containing the model and config files.
            device (torch.device): The device (CPU/GPU) to run the model on.
        """
        self.device = device
        self.model_dir = model_dir
        self.configs = load_config(f"{model_dir}/config.yaml")
        self.sample_rate = self.configs["sample_rate"]
        self._initialize_inference()

    def _initialize_inference(self):
        """Initializes the tokenizer, model, and audio tokenizer for inference."""
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_dir}/LLM")
        self.model = AutoModelForCausalLM.from_pretrained(f"{self.model_dir}/LLM")
        self.audio_tokenizer = BiCodecTokenizer(self.model_dir, device=self.device)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    def process_prompt(
        self,
        text: str,
        prompt_speech_path: Path,
        prompt_text: str = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        Process input for voice cloning.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.

        Return:
            Tuple[str, torch.Tensor]: Input prompt; global tokens
        """
        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(
            prompt_speech_path
        )
        global_tokens = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
        )

        # Prepare the input tokens for the model
        if prompt_text is not None:
            semantic_tokens = "".join(
                [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
            )
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                prompt_text,
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        else:
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]

        inputs = "".join(inputs)
        return inputs, global_token_ids

    def process_prompt_control(
        self,
        gender: str,
        pitch: str,
        speed: str,
        text: str,
    ) -> str:
        """
        Process input for voice creation.

        Args:
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            text (str): The text input to be converted to speech.

        Return:
            str: Input prompt
        """
        assert gender in GENDER_MAP.keys()
        assert pitch in LEVELS_MAP.keys()
        assert speed in LEVELS_MAP.keys()

        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        attribte_tokens = "".join(
            [gender_tokens, pitch_label_tokens, speed_label_tokens]
        )

        control_tts_inputs = [
            TASK_TOKEN_MAP["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            attribte_tokens,
            "<|end_style_label|>",
        ]

        return "".join(control_tts_inputs)

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
        overlap_duration: float = 0.1,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 3000,
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
            overlap_duration (float): Duration of overlap between chunks in seconds
            temperature (float): Sampling temperature for controlling randomness.
            top_k (int): Top-k sampling parameter.
            top_p (float): Top-p (nucleus) sampling parameter.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Yields:
            np.ndarray: Audio chunks as they are generated
        """
        # Prepare input prompt
        if gender is not None:
            prompt = self.process_prompt_control(gender, pitch, speed, text)
            # For control mode, we'll extract global tokens from the generated text
            global_token_ids = None
        else:
            prompt, global_token_ids = self.process_prompt(
                text, prompt_speech_path, prompt_text
            )
        
        # Tokenize input
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        
        # Calculate overlap samples
        overlap_samples = int(overlap_duration * self.sample_rate)
        
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
        
        # Create a custom generation config
        generation_config = self.model.generation_config
        generation_config.max_new_tokens = max_new_tokens
        generation_config.do_sample = True
        generation_config.temperature = temperature
        generation_config.top_k = top_k
        generation_config.top_p = top_p
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        
        # Generate tokens with callback
        # Note: Since transformers doesn't support streaming callbacks directly,
        # we'll use a custom generation loop
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask
        past_key_values = None
        
        for step in range(max_new_tokens):
            # Forward pass
            with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
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
                logits = torch.where(logits < min_top_k, 
                                   torch.tensor(-float('inf'), device=logits.device), 
                                   logits)
            
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
            
            # Check if it's a semantic token
            token_str = self.tokenizer.decode([next_token.item()])
            if "bicodec_semantic_" in token_str:
                match = re.search(r"bicodec_semantic_(\d+)", token_str)
                if match:
                    semantic_id = int(match.group(1))
                    generated_tokens.append(semantic_id)
                    
                    # Generate audio chunk when we have enough tokens
                    if len(generated_tokens) >= stream_chunk_size:
                        # Extract global tokens if needed
                        if gender is not None and global_token_ids is None:
                            generated_text = self.tokenizer.decode(all_generated_ids)
                            global_matches = re.findall(r"bicodec_global_(\d+)", generated_text)
                            if global_matches:
                                global_token_ids = torch.tensor(
                                    [[int(g) for g in global_matches]], 
                                    dtype=torch.long
                                ).unsqueeze(0).to(self.device)
                        
                        # Skip if we don't have global tokens yet
                        if global_token_ids is not None:
                            # Convert to tensor
                            semantic_ids = torch.tensor(
                                generated_tokens[:stream_chunk_size], 
                                dtype=torch.long
                            ).unsqueeze(0).to(self.device)
                            
                            # Generate audio for this chunk
                            wav_chunk = self.audio_tokenizer.detokenize(
                                global_token_ids.squeeze(0),
                                semantic_ids
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
            semantic_ids = torch.tensor(
                generated_tokens, 
                dtype=torch.long
            ).unsqueeze(0).to(self.device)
            
            wav_chunk = self.audio_tokenizer.detokenize(
                global_token_ids.squeeze(0),
                semantic_ids
            )
            
            # Final chunk doesn't need overlap handling
            yield wav_chunk


# Example usage
if __name__ == "__main__":
    import soundfile as sf
    
    # Initialize streaming model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparkTTSStreaming(
        model_dir="/raid/aluno_alef/Spark-TTS-finetune/checkpoints-spark-tts-multilingual",
        device=device
    )
    
    # Example 1: Voice cloning with streaming
    text = "Embora estivesse chovendo, eles decidiram passear na floresta."
    prompt_wav = "/raid/aluno_alef/Spark-TTS/ermis_11labs-0000-0003.wav"
    prompt_text = "A união faz a força, já dizia minha avó. E quando a gente se junta para resolver os problemas do bairro, as coisas fluem melhor."
    
    # Collect all chunks
    audio_chunks = []

    chunks = model.inference_stream(
        text=text,
        prompt_speech_path=prompt_wav,
        prompt_text=prompt_text,
        stream_chunk_size=20,  # Generate audio every 20 tokens
        overlap_duration=0.1   # 100ms overlap for smooth transitions
    )

    print(chunks)
    
    print("Generating audio stream...")
    for i, chunk in enumerate(model.inference_stream(
        text=text,
        prompt_speech_path=prompt_wav,
        prompt_text=prompt_text,
        stream_chunk_size=20,  # Generate audio every 20 tokens
        overlap_duration=0.1   # 100ms overlap for smooth transitions
    )):
        print(f"Generated chunk {i+1}, shape: {chunk.shape}")
        audio_chunks.append(chunk)
    
    # Concatenate all chunks
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        sf.write("streaming_output.wav", full_audio, 16000)
        print(f"Total audio length: {len(full_audio)/16000:.2f} seconds")
    
    # Example 2: Voice creation with streaming
    print("\nGenerating controlled voice stream...")
    audio_chunks = []
    
    for i, chunk in enumerate(model.inference_stream(
        text="This is a controlled voice generation test with specific parameters.",
        gender="female",
        pitch="high",
        speed="moderate",
        stream_chunk_size=15
    )):
        print(f"Generated chunk {i+1}, shape: {chunk.shape}")
        audio_chunks.append(chunk)
    
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        sf.write("streaming_controlled_output.wav", full_audio, 16000)
        print(f"Total audio length: {len(full_audio)/16000:.2f} seconds")