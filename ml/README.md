# build
docker build -t fashionclip-ml .


docker run --rm -v ${PWD}:/app fashionclip-ml python main.py test.jpg
