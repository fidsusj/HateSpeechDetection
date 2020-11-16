# HateSpeechDetection

Goal of this project is to classify and filter social media posts on hate speech using text analytics methods. 

## Categories

- Bullying
- Harassment
- Racism
- Sexism

## Datasets and papers

### hatespeechdata.com

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
    
### ACM (VPN required)

- [Deep Learning for Hate Speech Detection in Tweets](https://dl.acm.org/doi/pdf/10.1145/3041021.3054223)
- [Towards Automatic Detection and Explanation of Hate Speech and Offensive Language](https://dl.acm.org/doi/pdf/10.1145/3375708.3380312)
- [Detecting Hate, Offensive, and Regular Speech in Short Comments](https://dl.acm.org/doi/pdf/10.1145/3126858.3131576)
- [Mean Birds: Detecting Aggression and Bullying on Twitter](https://dl.acm.org/doi/pdf/10.1145/3091478.3091487)
- [A Unified Deep Learning Architecture for Abuse Detection](https://dl.acm.org/doi/pdf/10.1145/3292522.3326028)
- [Identifying Hate Speech in Social Media](https://dl.acm.org/doi/pdf/10.1145/3155212)

### IEEE

- [Hate Speech Detection on Twitter Using Long Short-Term Memory (LSTM) Method](https://ieeexplore.ieee.org/document/9003992) [$$$]
- [Automated Hate Speech Detection on Twitter](https://ieeexplore.ieee.org/document/9128428) [$$$]
- [Hate Speech on Twitter: A Pragmatic Approach to Collect Hateful and Offensive Expressions and Perform Hate Speech Detection](https://ieeexplore.ieee.org/document/8292838)
- [Hate Speech Detection on Twitter Using Multinomial Logistic Regression Classification Method](https://ieeexplore.ieee.org/document/8980379) [$$$]
- [A Framework for Hate Speech Detection using Deep Convolutional Neural Network](https://ieeexplore.ieee.org/document/9253658)
- [Text Analysis For Hate Speech Detection Using Backpropagation Neural Network](https://ieeexplore.ieee.org/document/8712109)
- [Analysis Text of Hate Speech Detection Using Recurrent Neural Network](https://ieeexplore.ieee.org/document/8712104)
- [Hate Speech Detection using Global Vector and Deep Belief Network Algorithm](https://ieeexplore.ieee.org/document/9245467)
- [Evaluating Machine Learning Techniques for Detecting Offensive and Hate Speech in South African Tweets](https://ieeexplore.ieee.org/document/8963960)

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