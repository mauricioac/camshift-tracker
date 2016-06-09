Trabalho de tracking e contagem de objetos com CamShift

Alunos: Silvana Trindade e Maurício André Cinelli

Importante: Código feito para ambiente Linux, testado em Ubuntu 14.04 e Debian 7.9. Versão da OpenCV 3.0!!

## Instrução para compilar

Vídeos utilizados foram os do moodle, porém convertidos para mp4. Para rodar os webm (que estão no moodle) é necessário ter os plugins da opencv instalados e configurados.

Para compilar, basta executar:

make

e para executar o programa:

./video [caminho_do_video]
ou
./video

O segundo irá utilizar o índice "1" da câmera do computador.

Ao executar, o programa irá capturar alguns frames para estabilizar o background.
Depois disso, o usuário deve selecionar com o mouse uma região para determinar o seguinte:

- o topo do quadrado selecionado é utilizado como ponto para capturar objetos, e iniciar o tracking dos mesmos
- a linha de baixo do quadrado selecionado é utilizada como ponto de referência para a contagem dos objetos (ponto de fuga dos objetos)

Para entender melhor como é este quadrado, veja a imagem "screenshot.png"
