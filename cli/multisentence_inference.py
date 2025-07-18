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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run TTS inference.")

    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="example/results",
        help="Directory to save generated audio files",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device number")
    parser.add_argument(
        "--text", type=str, help="Text for TTS generation"
    )
    parser.add_argument("--prompt_text", type=str, help="Transcript of prompt audio")
    parser.add_argument(
        "--prompt_speech_path",
        type=str,
        help="Path to the prompt audio file",
    )
    parser.add_argument("--gender", choices=["male", "female"])
    parser.add_argument(
        "--pitch", choices=["very_low", "low", "moderate", "high", "very_high"]
    )
    parser.add_argument(
        "--speed", choices=["very_low", "low", "moderate", "high", "very_high"]
    )
    return parser.parse_args()


def run_tts(args):
    """Perform TTS inference and save the generated audio."""
    logging.info(f"Using model from: {args.model_dir}")
    logging.info(f"Saving audio to: {args.save_dir}")

    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Convert device argument to torch.device
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        # macOS with MPS support (Apple Silicon)
        device = torch.device(f"mps:{args.device}")
        logging.info(f"Using MPS device: {device}")
    elif torch.cuda.is_available():
        # System with CUDA support
        device = torch.device(f"cuda:{args.device}")
        logging.info(f"Using CUDA device: {device}")
    else:
        # Fall back to CPU
        device = torch.device("cpu")
        logging.info("GPU acceleration not available, using CPU")

    # Initialize the model
    model = SparkTTS(args.model_dir, device)

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(args.save_dir, f"{timestamp}.wav")

    logging.info("Starting inference...")

    portuguese_sentences_short = [
        "Vai.",
        "Pare agora.",
        "Corra rápido.",
        "Pule alto.",
        "Coma comida.",
        "Beba água.",
        "Durma bem.",
        "Leia livros.",
        "Escreva código.",
        "Aprenda diariamente.",
        "Pense grande.",
        "Sorria sempre.",
        "Ria alto.",
        "Chore baixo.",
        "Ajude outros.",
        "Seja gentil.",
        "Fique calmo.",
        "Trabalhe duro.",
        "Jogue jogos.",
        "Divirta-se.",
        "Veja estrelas.",
        "Sinta alegria.",
        "Ouça música.",
        "Cheire flores.",
        "Viva a vida."
    ]

    portuguese_sentences_mid = [
        "O gato sentou-se no tapete.",
        "Ela caminhou rapidamente até a loja.",
        "Ele gosta de jogar basquete no parque.",
        "Pássaros cantam lindamente nas árvores.",
        "O sol está brilhando forte hoje.",
        "Eles estão planejando uma viagem para a Europa em breve.",
        "Precisamos comprar mantimentos para o jantar.",
        "Minha cor favorita é definitivamente azul.",
        "Aprender coisas novas é sempre emocionante.",
        "A casa antiga ficava em uma colina.",
        "A chuva está caindo suavemente lá fora agora.",
        "Café pela manhã é essencial para mim.",
        "Caminhar na natureza é muito tranquilo.",
        "Bons amigos são importantes na vida.",
        "A tecnologia está mudando rapidamente hoje em dia.",
        "A música pode expressar muitas emoções diferentes.",
        "Ler um bom romance é um grande prazer.",
        "Cozinhar refeições deliciosas é um hobby divertido.",
        "Explorar novas cidades é sempre interessante.",
        "Ter um animal de estimação pode trazer muita felicidade.",
        "Falar línguas diferentes é muito útil.",
        "Viajar para novos lugares amplia sua mente.",
        "Escrita criativa pode ser muito terapêutica.",
        "Resolver problemas complexos é bastante desafiador.",
        "Construir relacionamentos fortes leva tempo e esforço."
    ]

    portuguese_sentences_mid_long = [
        "Embora estivesse chovendo, eles decidiram passear na floresta.",
        "Por causa do trânsito intenso, chegamos à reunião um pouco atrasados.",
        "Mesmo se sentindo cansada, ela continuou a trabalhar em seu projeto até tarde da noite.",
        "Se você quer ter sucesso, deve estar preparado para trabalhar muito duro e manter o foco.",
        "Antes de sair de férias, lembre-se de regar as plantas e trancar todas as portas.",
        "Depois de terminar o trabalho, ele relaxou ouvindo música clássica e lendo um livro.",
        "Enquanto as crianças brincavam no jardim, seus pais preparavam o almoço na cozinha.",
        "Apesar das dificuldades, eles conseguiram concluir o projeto com sucesso e no prazo.",
        "Considerando a previsão do tempo, pode ser uma boa ideia levar um guarda-chuva com você hoje.",
        "Para melhorar suas habilidades de comunicação, pratique falar em público e ouça ativamente os outros.",
        "Para atingir seus objetivos, é importante definir metas realistas e criar um plano.",
        "Sempre que me sinto estressado, gosto de dar uma caminhada no parque ou ouvir música calmante.",
        "Assim que o filme começou, as luzes diminuíram e todos ficaram quietos no teatro.",
        "Como era feriado, muitas famílias decidiram passar o dia na praia ou no campo.",
        "Antes de tomar uma decisão, é sempre sensato considerar todas as opções e resultados possíveis.",
        "Embora nunca tivesse tentado antes, ele achou o snowboard surpreendentemente divertido e emocionante.",
        "Como ela era muito organizada e eficiente, sempre conseguia terminar suas tarefas rapidamente.",
        "Mesmo que o restaurante estivesse lotado, eles encontraram uma mesa agradável perto da janela com uma ótima vista.",
        "Se você está planejando viajar para o exterior, é essencial ter um passaporte válido e quaisquer vistos necessários.",
        "Antes de iniciar qualquer novo programa de exercícios, é aconselhável consultar seu médico ou um especialista em fitness.",
        "Depois de passar muitos anos morando na cidade, eles decidiram se mudar para uma vila tranquila e pacífica.",
        "Enquanto o chef preparava o prato principal, os garçons estavam preparando as mesas para a noite.",
        "Apesar dos contratempos iniciais, a equipe perseverou e acabou atingindo seus objetivos de longo prazo.",
        "Considerando a complexidade da situação, é importante abordar o problema com planejamento e estratégia cuidadosos.",
        "Para garantir o sucesso do evento, precisamos coordenar todos os diferentes aspectos e trabalhar juntos de forma eficaz."
    ]

    portuguese_sentences_long = [
        "No reino da física teórica, os cientistas estão constantemente explorando os mistérios do universo, desde as menores partículas subatômicas até as maiores estruturas cósmicas, buscando entender as leis fundamentais que governam a realidade e as intrincadas relações entre espaço, tempo, matéria e energia, muitas vezes empregando modelos matemáticos complexos e experimentos mentais para sondar as fronteiras do conhecimento humano e impulsionar as fronteiras da compreensão científica.",
        "O desenvolvimento da inteligência artificial, com seu potencial para revolucionar vários aspectos da vida humana, desde saúde e transporte até comunicação e entretenimento, levanta profundas questões éticas e sociais sobre o futuro do trabalho, a natureza da consciência e os potenciais riscos e benefícios de máquinas cada vez mais autônomas, exigindo cuidadosa consideração e medidas proativas para garantir inovação responsável e distribuição equitativa de seu poder transformador em diferentes comunidades e demografias.",
        "A sustentabilidade ambiental, um desafio crítico que a humanidade enfrenta no século XXI, exige um esforço global e colaborativo para abordar as mudanças climáticas, a perda de biodiversidade e a depleção de recursos por meio de uma combinação de inovação tecnológica, mudanças políticas e ações individuais, visando desvincular o crescimento econômico da degradação ambiental e transitar para uma economia circular e regenerativa que priorize a integridade ecológica, a equidade social e o bem-estar a longo prazo para as gerações presentes e futuras.",
        "O estudo da história humana, abrangendo diversas culturas, civilizações e sociedades ao longo de milênios, fornece insights valiosos sobre a jornada complexa e multifacetada da humanidade, destacando tanto conquistas notáveis quanto fracassos trágicos, oferecendo lições sobre a natureza cíclica do poder, as lutas duradouras por justiça e igualdade e a interação constante entre continuidade e mudança que molda a trajetória dos assuntos humanos, sublinhando a importância de entender o passado para navegar no presente e vislumbrar um futuro mais justo e sustentável.",
        "A globalização, caracterizada pela crescente interconexão de nações e sociedades por meio do comércio, migração, comunicação e intercâmbio cultural, apresenta oportunidades e desafios para indivíduos, comunidades e governos em todo o mundo, fomentando o crescimento econômico e a inovação, ao mesmo tempo em que exacerba as desigualdades, cria novas formas de vulnerabilidade e levanta questões complexas sobre soberania nacional, identidade cultural e a governança de bens comuns globais, exigindo cooperação internacional e políticas inclusivas para aproveitar seus benefícios e mitigar seus riscos de forma a promover prosperidade compartilhada e bem-estar planetário.",
        "A exploração do espaço profundo, impulsionada pela curiosidade humana e pela busca de entender nosso lugar no cosmos, representa um esforço grandioso e ambicioso que ultrapassa os limites da engenhosidade humana e das capacidades tecnológicas, prometendo não apenas descobertas científicas inovadoras sobre as origens do universo, o potencial de vida além da Terra e a vastidão dos fenômenos cósmicos, mas também inspirando as gerações futuras, fomentando a colaboração internacional e potencialmente desbloqueando recursos e oportunidades que poderiam beneficiar a humanidade de maneiras profundas e imprevistas, embora exigindo investimentos substanciais e consideração cuidadosa das implicações éticas e ambientais.",
        "Avanços na ciência médica, desde o desenvolvimento de novas vacinas e terapias para doenças infecciosas e doenças crônicas até o refinamento de técnicas cirúrgicas e ferramentas de diagnóstico, melhoraram drasticamente a saúde humana e a longevidade no século passado, levando ao aumento da expectativa de vida, à redução da mortalidade infantil e à melhoria da qualidade de vida para milhões, mas também colocando dilemas éticos complexos relacionados ao acesso aos cuidados de saúde, ao custo dos tratamentos e à distribuição equitativa de recursos médicos em diferentes populações e regiões, exigindo pesquisa contínua, debates políticos e colaborações globais para garantir que os benefícios do progresso médico sejam compartilhados por todos.",
        "O poder transformador da educação, abrangendo a escolaridade formal, o aprendizado ao longo da vida e o acesso ao conhecimento e recursos de informação, desempenha um papel crucial no empoderamento individual, na mobilidade social e no desenvolvimento econômico, equipando os indivíduos com as habilidades, competências e habilidades de pensamento crítico necessárias para prosperar em um mundo cada vez mais complexo e em rápida mudança, promovendo a criatividade, a inovação e o engajamento cívico, e contribuindo para a criação de sociedades mais inclusivas, equitativas e democráticas, sublinhando a importância de investir em educação de qualidade para todos e garantir oportunidades iguais de aprendizagem e crescimento pessoal, independentemente da origem ou circunstância.",
        "A intrincada teia de ecossistemas, abrangendo florestas, oceanos, pastagens e desertos, desempenha um papel vital na manutenção do equilíbrio ecológico do planeta, na regulação dos padrões climáticos, no fornecimento de recursos e serviços essenciais para as sociedades humanas e no suporte a uma rica tapeçaria de biodiversidade, enfrentando ameaças crescentes de destruição de habitat, poluição, superexploração e mudanças climáticas, exigindo esforços de conservação urgentes e concertados para proteger e restaurar esses sistemas naturais, garantindo seu funcionamento e resiliência contínuos para o benefício das gerações presentes e futuras e reconhecendo o valor intrínseco de todos os organismos vivos e a interconexão da vida na Terra.",
        "A rápida proliferação de tecnologias digitais, incluindo a internet, dispositivos móveis, plataformas de mídia social e sistemas de inteligência artificial, remodelou profundamente a comunicação, o comércio, a cultura e as interações sociais em todo o mundo, criando oportunidades sem precedentes para compartilhamento de informações, colaboração e crescimento econômico, ao mesmo tempo em que levanta preocupações sobre privacidade, segurança, desinformação, viés algorítmico e o potencial para que as divisões digitais exacerbarem as desigualdades existentes, exigindo políticas ponderadas, estruturas éticas e soluções tecnológicas para aproveitar os benefícios da inovação digital, mitigando seus riscos e garantindo que a tecnologia sirva a humanidade de uma forma que seja empoderadora e equitativa para todos os membros da sociedade.",
        "A mecânica quântica, a teoria que descreve as propriedades físicas da natureza na escala de átomos e partículas subatômicas, sustenta nossa compreensão dos blocos de construção fundamentais da matéria e das forças que governam suas interações, revelando um mundo contraintuitivo e probabilístico onde as partículas podem existir em múltiplos estados simultaneamente e exibir dualidade onda-partícula, desafiando as noções clássicas de determinismo e localidade e abrindo caminho para tecnologias revolucionárias, como a computação quântica e a criptografia quântica, que prometem resolver problemas atualmente intratáveis e transformar o processamento de informações.",
        "O cérebro humano, um órgão complexo e altamente evoluído composto por bilhões de neurônios interconectados, serve como o centro de controle do sistema nervoso, responsável pela percepção, cognição, emoção e comportamento, permitindo-nos aprender, lembrar, raciocinar e interagir com o mundo ao nosso redor, e embora muito progresso tenha sido feito em neurociência, muitos mistérios permanecem sobre seu funcionamento intrincado, incluindo a base neural da consciência, os mecanismos de aprendizagem e memória e as causas e tratamentos de distúrbios neurológicos e psiquiátricos, impulsionando a pesquisa e a inovação contínuas destinadas a desvendar as complexidades do cérebro e melhorar as habilidades cognitivas humanas.",
        "A vastidão do universo, contendo bilhões de galáxias, cada uma com bilhões de estrelas, estende-se muito além de nossas capacidades de observação atuais, com pesquisas astronômicas contínuas e missões espaciais continuamente ultrapassando os limites de nosso conhecimento sobre o cosmos, revelando a existência de objetos exóticos, como buracos negros, estrelas de nêutrons e quasares, e descobrindo novos insights sobre a formação e evolução de galáxias, a distribuição de matéria escura e energia escura e o potencial de vida além da Terra, inspirando admiração e espanto ao mesmo tempo em que levanta questões fundamentais sobre nosso lugar no grande esquema cósmico e o destino final do universo.",
        "As mudanças climáticas, impulsionadas principalmente por atividades humanas, como a queima de combustíveis fósseis e o desmatamento, representam uma ameaça significativa aos ecossistemas do planeta e às sociedades humanas, levando ao aumento das temperaturas globais, elevação do nível do mar, eventos climáticos extremos mais frequentes e intensos e interrupções nos sistemas agrícolas e recursos hídricos, exigindo estratégias urgentes e abrangentes de mitigação e adaptação para reduzir as emissões de gases de efeito estufa, transitar para fontes de energia renováveis, proteger as populações vulneráveis e construir resiliência aos impactos de um clima em mudança, garantindo um futuro sustentável e habitável para todos.",
        "O poder da música, transcendendo fronteiras culturais e diferenças linguísticas, tem sido reconhecido ao longo da história humana como uma linguagem universal capaz de evocar emoções profundas, promover a coesão social e melhorar as habilidades cognitivas, com diversas tradições e gêneros musicais refletindo a rica tapeçaria da criatividade e expressão humanas, servindo como fonte de inspiração, entretenimento e cura e desempenhando um papel vital na identidade cultural, práticas rituais e movimentos sociais, ressaltando seu significado duradouro na vida humana e seu potencial para superar divisões e promover a compreensão entre culturas e comunidades.",
        "A importância da biodiversidade, abrangendo a variedade de vida na Terra em todos os níveis, de genes a ecossistemas, reside em seu papel crucial na manutenção do funcionamento do ecossistema, fornecendo serviços ecossistêmicos essenciais, como polinização, purificação da água e regulação do clima e apoiando o bem-estar humano e a prosperidade econômica, enfrentando ameaças sem precedentes de perda de habitat, espécies invasoras, poluição e mudanças climáticas, exigindo esforços de conservação concertados para proteger e restaurar a biodiversidade, garantindo a saúde e a resiliência a longo prazo de sistemas naturais e humanos e salvaguardando a herança insubstituível da vida na Terra para as gerações futuras.",
        "As considerações éticas no desenvolvimento da inteligência artificial estão se tornando cada vez mais críticas à medida que os sistemas de IA se tornam mais sofisticados e integrados em vários aspectos da vida humana, levantando preocupações sobre viés algorítmico, deslocamento de empregos, violações de privacidade, armas autônomas e o potencial de consequências não intencionais, exigindo deliberação cuidadosa e o estabelecimento de diretrizes éticas e estruturas regulatórias para garantir que a IA seja desenvolvida e implantada de forma responsável, ética e de forma que beneficie toda a humanidade, promovendo justiça, transparência e responsabilidade em sistemas de IA e mitigando riscos potenciais.",
        "Os avanços contínuos em biotecnologia e engenharia genética têm um imenso potencial para revolucionar a medicina, a agricultura e a indústria, oferecendo novas ferramentas para tratar doenças, aumentar a produção de culturas e desenvolver materiais sustentáveis, ao mesmo tempo em que levantam preocupações éticas e sociais sobre modificação genética, biossegurança e o potencial de consequências ecológicas e de saúde não intencionais, exigindo avaliação de risco cuidadosa, diálogo público e inovação responsável para aproveitar os benefícios da biotecnologia, mitigando os riscos potenciais e garantindo acesso equitativo a seus benefícios e salvaguardas contra uso indevido e danos não intencionais.",
        "A exploração da consciência, um mistério fundamental tanto na neurociência quanto na filosofia, busca entender a experiência subjetiva da consciência, a natureza da qualia e a relação entre mente e cérebro, levantando questões profundas sobre as origens da autoconsciência, a possibilidade de consciência artificial e as implicações éticas da criação de máquinas sencientes, impulsionando a pesquisa e o debate contínuos em várias disciplinas e ultrapassando os limites de nossa compreensão do que significa ser consciente e o lugar único da consciência no universo, com implicações para a ciência e para nossa compreensão da condição humana.",
        "O significado do patrimônio cultural, abrangendo formas tangíveis e intangíveis de expressão cultural, incluindo monumentos, tradições, línguas e práticas artísticas, reside em seu papel na formação da identidade, na promoção da coesão social e na transmissão de valores e conhecimentos entre gerações, enfrentando ameaças da globalização, conflitos e degradação ambiental, exigindo esforços concertados para proteger e preservar o patrimônio cultural, garantindo sua acessibilidade às gerações futuras e reconhecendo seu valor intrínseco como fonte de criatividade humana, diversidade e história compartilhada e sua contribuição para enriquecer vidas humanas e promover a compreensão e o respeito interculturais.",
        "Os desafios da urbanização, com mais da metade da população mundial vivendo agora em cidades, apresentam oportunidades e problemas para o desenvolvimento sustentável, populações urbanas concentradas podem promover a inovação, o crescimento econômico e a diversidade cultural, ao mesmo tempo em que sobrecarregam a infraestrutura, exacerbam as desigualdades e contribuem para a poluição ambiental e a depleção de recursos, exigindo planejamento urbano integrado, desenvolvimento de infraestrutura sustentável e governança inclusiva para criar cidades habitáveis, resilientes e equitativas que possam acomodar populações crescentes, minimizando o impacto ambiental e maximizando a qualidade de vida para todos os residentes, garantindo um futuro urbano sustentável.",
        "O poder da narrativa, um aspecto fundamental da comunicação e cultura humanas, tem sido usado por milênios para transmitir conhecimento, valores e crenças entre gerações, para entreter, inspirar e conectar pessoas e para dar sentido ao mundo ao nosso redor, com diversas formas de narrativa, desde tradições orais e mitos até literatura, cinema e mídia digital, servindo como uma ferramenta poderosa para empatia, compreensão e mudança social e desempenhando um papel vital na formação de identidades individuais e coletivas, promovendo o intercâmbio cultural e promovendo a compreensão e a compaixão humanas em diversas comunidades e sociedades.",
        "O papel crítico da cooperação internacional no enfrentamento de desafios globais, como mudanças climáticas, pandemias, pobreza e desigualdade, é cada vez mais reconhecido como essencial para alcançar o desenvolvimento sustentável e garantir um futuro pacífico e próspero para todos, exigindo multilateralismo, diplomacia e ação coletiva para enfrentar ameaças compartilhadas, promover objetivos comuns e construir uma ordem mundial mais justa e equitativa, exigindo colaboração entre nações, culturas e setores para lidar com questões globais complexas de forma eficaz e para promover um senso de cidadania global e responsabilidade compartilhada pelo bem-estar da humanidade e do planeta, transcendendo fronteiras nacionais e promovendo a solidariedade global.",
        "A exploração do oceano profundo, um reino vasto e em grande parte inexplorado que cobre a maioria da superfície da Terra, guarda imenso potencial para descobertas científicas, exploração de recursos e compreensão do intrincado funcionamento dos ecossistemas marinhos, com expedições de pesquisa contínuas e avanços tecnológicos continuamente revelando novas espécies, descobrindo fontes hidrotermais e trincheiras de profundidade e lançando luz sobre o papel crucial do oceano na regulação do clima, no suporte à biodiversidade e no fornecimento de recursos para as sociedades humanas, ao mesmo tempo em que levanta preocupações sobre a poluição marinha, a pesca excessiva e a necessidade de gestão oceânica sustentável para proteger esta parte vital e muitas vezes esquecida do nosso planeta para as gerações futuras.",
        "A interseção de tecnologia e sociedade apresenta desafios e oportunidades complexas e em evolução à medida que as tecnologias digitais se tornam cada vez mais generalizadas e integradas em todos os aspectos de nossas vidas, exigindo um exame crítico das implicações sociais, éticas e políticas dos avanços tecnológicos, incluindo questões de privacidade, segurança, viés algorítmico, desinformação e exclusão digital e exigindo abordagens ponderadas e proativas para moldar o desenvolvimento e a implantação da tecnologia de forma a se alinhar com os valores humanos, promover a justiça social e promover um futuro mais inclusivo, equitativo e sustentável para todos os membros da sociedade, garantindo que a tecnologia sirva à humanidade e não o contrário."
    ]

    lists_of_sentences = [
        # ("short", portuguese_sentences_short),
        ("mid", portuguese_sentences_mid),
        ("mid_long", portuguese_sentences_mid_long),
        # ("long", portuguese_sentences_long)
    ]

    short_times = []
    mid_times = []
    mid_long_times = []
    long_times = []

    durations = []
    for length, sentences in tqdm(lists_of_sentences):
        print(length)
        print(sentences)
        print(f"Generating {length} sentences...")
        for idx, sentence in tqdm(enumerate(sentences)):
            output_path = os.path.join(args.save_dir, f"{length}_{length}_{idx}.wav")
            # Perform inference and save the output audio
            start_time = time.time()
            with torch.no_grad():
                try:
                    wav = model.inference(
                        sentence,
                        args.prompt_speech_path,
                        # prompt_text=args.prompt_text,
                        gender=args.gender,
                        pitch=args.pitch,
                        speed=args.speed,
                    )
                except Exception as e:
                    logging.error(f"Error generating audio for sentence '{sentence}': {e}")
                    continue
                dur = wav.shape[-1] / 16000
                print(dur)
                durations.append(dur)
                sf.write(output_path, wav, samplerate=16000)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(dur, elapsed_time)
            if length == "short":
                short_times.append(elapsed_time)
            elif length == "mid":
                mid_times.append(elapsed_time)
            elif length == "mid_long":
                mid_long_times.append(elapsed_time)
            elif length == "long":
                long_times.append(elapsed_time)
    # Avg times
    avg_short_time = sum(short_times) / len(short_times)
    avg_mid_time = sum(mid_times) / len(mid_times)
    avg_mid_long_time = sum(mid_long_times) / len(mid_long_times)
    avg_long_time = sum(long_times) / len(long_times)

    print(f"Avg time for short sentences: {avg_short_time:.2f} seconds")
    print(f"Avg time for mid sentences: {avg_mid_time:.2f} seconds")
    print(f"Avg time for mid_long sentences: {avg_mid_long_time:.2f} seconds")
    logging.info(f"Avg time for long sentences: {avg_long_time:.2f} seconds")

    rtf = []
    for duration, run_time in zip(durations, short_times):
        rtf.append(duration / run_time)
    avg_rtf = sum(rtf) / len(rtf)
    print(f"Avg RTF for short sentences: {avg_rtf:.3f}")

    # calculate RTF
    rtf = []
    for duration, run_time in zip(durations, mid_times):
        rtf.append(duration / run_time)
    avg_rtf = sum(rtf) / len(rtf)

    print(f"Avg RTF for mid sentences: {avg_rtf:.3f}")

    rtf = []
    for duration, run_time in zip(durations, mid_long_times):
        rtf.append(duration / run_time)
    avg_rtf = sum(rtf) / len(rtf)
    print(f"Avg RTF for mid_long sentences: {avg_rtf:.3f}")

    rtf = []
    for duration, run_time in zip(durations, long_times):
        rtf.append(duration / run_time)
    avg_rtf = sum(rtf) / len(rtf)
    print(f"Avg RTF for long sentences: {avg_rtf:.3f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_args()
    run_tts(args)
