# pln-classificador
Classificador de texto utilizado para a classe de Processamento de Linguagem Natural em 2015.2 no CIn.

Como utilizar o parser 'sgml_parser.py': Esse arquivo contém uma função chamanda get_documents_from_sgml(). Ela é responsável por ler os arquivos que recebemos como base de dados, que estão na extensão .sgml, e transformá-los em um dicionário de documentos.

Cada documento está no dicionário de acordo com seu ID. Por exemplo, se chamarmos o dicionário retornado por essa função de documents, então documents['0'] vai retornar informações sobre o documento de id '0' na nossa base.

Cada índice do mapa documents, como dito anteriormente, traz informações sobre o documento correspondente. Essas informações podem ser os tópicos, que tem chave 'topics', se é de treinamento ou de teste, que tem chave 'split', e também o texto. O texto pode ter várias subseções, como data, autor, título e corpo. Portanto, cada documento tem uma chava 'text', que tem subchaves, como 'dateline', 'title', 'body', entre outros.

Como um exemplo curto, get_documents_from_sgml()['3']['topics'] vai nos retornar a lista de tópicos daquele documento. 
