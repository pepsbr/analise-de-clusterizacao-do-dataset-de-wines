# Análise de Clusterização do Dataset Wine

Projeto acadêmico de análise exploratória e clusterização hierárquica do conjunto **Wine**, disponibilizado pelo UCI Machine Learning Repository. O objetivo é explorar semelhanças entre vinhos descritos por atributos químicos e comparar agrupamentos com as classes originais do dataset.

## Conteúdo

- `Atividade 01 - Cluesterização.ipynb`: obtenção dos dados, exploração, inspeção de outliers, padronização, comparação de distâncias, clusterização hierárquica e visualização dos resultados.

## Dados

O notebook obtém os dados diretamente do UCI pelo pacote `ucimlrepo` (dataset ID 109). A versão usada contém 178 observações, 13 atributos químicos e a classe original (`target`). Os dados não estão duplicados neste repositório; eles são carregados durante a execução do notebook.

Fonte: [UCI Machine Learning Repository — Wine](https://archive.ics.uci.edu/dataset/109/wine).

## Métodos abordados

- Exploração de valores ausentes, duplicatas e outliers;
- Padronização dos atributos para colocá-los em escalas comparáveis;
- Comparação entre distâncias Euclidiana e Manhattan;
- Agrupamento hierárquico com ligação média (`average`) e inspeção do dendrograma;
- Comparação dos clusters com as classes originais e visualização em duas dimensões.

A escolha de três grupos é exploratória e considera a estrutura observada no dendrograma e as três classes conhecidas do dataset. Ela não deve ser interpretada como evidência de causalidade ou como uma avaliação independente de qualidade preditiva.

## Como executar

1. Instale Python 3.10 ou superior.
2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Abra o notebook no Jupyter ou Google Colab e execute as células na ordem. A primeira etapa instala/usa `ucimlrepo` e busca o dataset no UCI; é necessário acesso à internet.

## Créditos

Trabalho da Sala 4. Consulte a primeira seção do notebook para os créditos completos da equipe.
