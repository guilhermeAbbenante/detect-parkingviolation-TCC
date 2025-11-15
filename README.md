# 🚗 Serviço de Detecção de Infrações de Estacionamento

Este módulo é um serviço de visão computacional, parte do projeto [back-tcc-unip](https://github.com/Hitalo-27/back-tcc-unip), focado em analisar imagens e detectar veículos cometendo infrações de estacionamento.

Ele utiliza um modelo de segmentação de instâncias (YOLOv11) para identificar carros e zonas proibidas, aplicando uma lógica de regras para determinar se há uma infração.

## 🛠️ Tecnologias Utilizadas

* **Python 3.x**
* **Ultralytics (YOLOv11):** Para o modelo de detecção e segmentação.
* **OpenCV:** Para manipulação de imagens e processamento visual.
* **NumPy:** Para cálculos de máscara e interseção.

## 🧠 Como Funciona

O script carrega o modelo treinado (`best.pt`) e processa imagens de entrada. A lógica de infração é dividida em duas categorias:

1.  **Infração por Sobreposição (Interseção):**
    O script verifica se a máscara de pixels de um `carro` se sobrepõe (acima de um limite) com as máscaras de zonas proibidas, como:
    * `calcada`
    * `faixa_pedestre`
    * `guia_amarela` (meio-fio amarelo)
    * `guia_rebaixada` (entrada de garagem)
    * `rampa` (rampa de acessibilidade)

2.  **Infração Relacional (Proximidade):**
    O script verifica se um `carro` está estacionado muito próximo de uma `placa_proibido` (placa de proibido estacionar). Isso é feito analisando a distância entre os centros dos objetos e a proporção de seus tamanhos.

O script então gera novas imagens nas pastas `resultados/` ou `resultadoUnico/`, destacando os veículos, as zonas proibidas e o tipo de infração detectada.

## 🚀 Como Executar

**Importante:** Este projeto usa um `.gitignore` que ignora arquivos de modelo (`.pt`) e pastas de resultados (`resultadoUnico/`, `resultados/`). Você precisará adicionar o modelo manualmente.

1.  **Instalar Dependências:**
    Certifique-se de que está no diretório `servico_deteccao_vagas/` e instale as bibliotecas necessárias.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Adicionar o Modelo:**
    Obtenha o arquivo do modelo treinado (ex: `best.pt`) e coloque-o dentro da pasta `model/`.
    ```
    servico_deteccao_vagas/
    └── model/
        └── best.pt  <-- ADICIONE O MODELO AQUI
    ```

3.  **Executar o Script:**
    Você tem duas opções para executar o processamento:

    ### Opção A: Testar Múltiplas Imagens (em lote)
    
    O script **`mains.py`** (com 's' no final) processa **todas** as imagens da pasta `imagens/` e salva os resultados em `resultados/`.

    * **Execução:**
        ```bash
        python mains.py
        ```
    * **Resultado:** Os arquivos processados serão salvos na pasta `resultados/`.

    ### Opção B: Testar uma Única Imagem
    
    O script **`main.py`** (sem 's') processa **apenas uma** imagem específica e salva o resultado em `resultadoUnico/`.

    * **Configuração (se necessário):**
        Abra o `main.py` e altere a variável `SOURCE_IMAGE_PATH` para apontar para a imagem que você deseja testar.
        ```python
        # Dentro de main.py
        SOURCE_IMAGE_PATH = 'imagens/sua-imagem-de-teste.jpg' 
        ```
    * **Execução:**
        ```bash
        python main.py
        ```
    * **Resultado:** A imagem processada será salva na pasta `resultadoUnico/`.