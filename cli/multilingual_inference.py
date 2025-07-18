# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import torch
from tqdm import tqdm
import soundfile as sf
import logging
from datetime import datetime
import platform

from cli.SparkTTS import SparkTTS


def main():
    model_dir = "/raid/aluno_alef/Spark-TTS-finetune/checkpoints"
    save_dir = "output/multilingual_inference_v5-121k-ermis"

    prompt_text="A união faz a força, já dizia minha avó. E quando a gente se junta para resolver os problemas do bairro, as coisas fluem melhor."
    prompt_speech_path="/raid/aluno_alef/Spark-TTS/ermis_11labs-0000-0003.wav"

    # prompt_text="So maybe, that you would prefer to forgo my secret rather than consent to becoming a prisoner here for what might be several days."
    # prompt_speech_path="/raid/aluno_alef/Spark-TTS/sherlock_prompt.wav"

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda:0")
    print(f"Using CUDA device: {device}")

    # Initialize the model
    model = SparkTTS(model_dir, device)

    sentences = [
        ("pt", "Embora estivesse chovendo, eles decidiram passear na floresta."),
        ("pt", "Por causa do trânsito intenso, chegamos à reunião um pouco atrasados."),
        ("pt", "Mesmo se sentindo cansada, ela continuou a trabalhar em seu projeto até tarde da noite."),
        ("pt", "Se você quer ter sucesso, deve estar preparado para trabalhar muito duro e manter o foco."),
        ("pt", "Antes de sair de férias, lembre-se de regar as plantas e trancar todas as portas."),

        ("en", "Although it was raining, they decided to go for a walk in the forest."),
        ("en", "Because of the heavy traffic, we arrived at the meeting slightly late."),
        ("en", "Even though she felt tired, she continued to work on her project until late night."),
        ("en", "If you want to succeed, you must be prepared to work very hard and stay focused."),
        ("en", "Before leaving for vacation, remember to water the plants and lock all the doors."),

        ("it", "Sebbene piovesse, hanno deciso di fare una passeggiata nella foresta."),
        ("it", "A causa del traffico intenso, siamo arrivati alla riunione leggermente in ritardo."),
        ("it", "Anche se si sentiva stanca, ha continuato a lavorare al suo progetto fino a tarda notte."),
        ("it", "Se vuoi avere successo, devi essere pronto a lavorare molto duramente e rimanere concentrato."),
        ("it", "Prima di partire per le vacanze, ricordati di innaffiare le piante e chiudere a chiave tutte le porte."),

        ("pl", "Chociaż padało, postanowili wybrać się na spacer po lesie."),
        ("pl", "Z powodu dużego natężenia ruchu drogowego dotarliśmy na spotkanie nieco spóźnieni."),
        ("pl", "Mimo że była zmęczona, pracowała nad swoim projektem do późna w nocy."),
        ("pl", "Jeśli chcesz odnieść sukces, musisz być gotowy ciężko pracować i pozostać skupionym."),
        ("pl", "Przed wyjazdem na wakacje pamiętaj, by podlać rośliny i zamknąć wszystkie drzwi na klucz."),

        ("es", "Aunque llovía, decidieron dar un paseo por el bosque."),
        ("es", "Debido al intenso tráfico, llegamos un poco tarde a la reunión."),
        ("es", "A pesar de que se sentía cansada, siguió trabajando en su proyecto hasta altas horas de la noche."),
        ("es", "Si quieres tener éxito, debes estar dispuesto a trabajar muy duro y mantener la concentración."),
        ("es", "Antes de salir de vacaciones, recuerda regar las plantas y cerrar con llave todas las puertas."),

        ("fr", "Malgré la pluie, ils ont décidé de faire une promenade en forêt."),
        ("fr", "À cause de la circulation dense, nous sommes arrivés légèrement en retard à la réunion."),
        ("fr", "Même si elle était fatiguée, elle a continué à travailler sur son projet jusqu’à tard dans la nuit."),
        ("fr", "Si tu veux réussir, tu dois être prêt à travailler très dur et à rester concentré."),
        ("fr", "Avant de partir en vacances, n’oublie pas d’arroser les plantes et de fermer toutes les portes à clé."),
    ]

    for idx, (language, sentence) in tqdm(enumerate(sentences)):
        print(language)
        print(sentence)
        output_path = os.path.join(save_dir, f"{language}_{idx}.wav")
        with torch.no_grad():
            try:
                wav = model.inference(
                    sentence,
                    prompt_speech_path,
                    prompt_text=prompt_text,
                    gender=None,
                    pitch=None,
                    speed=None,
                    language=language,
                    use_multilingual=True,
                )
            except Exception as e:
                logging.error(f"Error generating audio for sentence '{sentence}': {e}")
                continue
            sf.write(output_path, wav, samplerate=16000)


if __name__ == "__main__":
    main()
