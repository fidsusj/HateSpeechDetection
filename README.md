# HateSpeechDetection

Goal of this project is to classify and filter social media posts on hate speech using text analytics methods. 

## Categories

- Bullying
- Harassment
- Racism
- Sexism

## Datasets and papers

- Cyberbullying Datasets (WoW Forum and LoL Forum)
    - Datasets: [Link](http://ub-web.de/research/)
    - Paper: [Link](https://aisel.aisnet.org/cgi/viewcontent.cgi?article=1061&context=ecis2016_rp)
- Hate Speech and Offensive Content Identification (HASOC)
    - Datasets: [Link](https://hasocfire.github.io/hasoc/2019/dataset.html)
    - Paper: [Link](https://dl.acm.org/doi/abs/10.1145/3368567.3368584?download=true)
- Hate Speech Twitter
    - Datasets: [Link](https://github.com/mayelsherif/hate_speech_icwsm18)
    - Paper: [Link](https://arxiv.org/pdf/1909.04251.pdf)
- Fox-News-User-Comments
    - Datasets: [Link](https://github.com/sjtuprog/fox-news-comments)
    - Paper: [Link](https://arxiv.org/pdf/1710.07395.pdf)

Taken from http://hatespeechdata.com/

## Commands

1. Install pipenv

       pip install pipenv
        
2. Install all the dependencies defined in the Pipfile        
        
       pipenv install --dev
        
3. Enter the virtual environment of pipenv

       pipenv shell

4. Set up the git hook scripts
       
       pre-commit install

5. Run the program

       pipenv run main
       
6. Run the tests

       pipenv run test && pipenv run report
       
7. Leave the virtual environment of pipenv

       exit