# Bot-and-Gender-Detection-of-Twitter-Accounts
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![License: GPL v3](https://img.shields.io/badge/-Sapienza%20University%20of%20Rome-red)](https://www.gnu.org/licenses/gpl-3.0)
# Citation 

To cite this work please use:
```
@article{bacciu2019bot,
  title={Bot and Gender Detection of Twitter Accounts Using Distortion and LSA},
  author={Bacciu, Andrea and La Morgia, Massimo and Mei, Alessandro and Nemmi, Eugenio Nerio and Neri, Valerio and Stefa, Julinda},
  year={2019}
}
```
[Link to the original paper @ ceur-ws.org](http://ceur-ws.org/Vol-2380/paper_210.pdf)  <br>

# Abstract
In this work, we present our approach for the Author Profiling task of PAN 2019.
The task is divided into two sub-problems, bot, and gender detection, for two different languages: English and Spanish.  For each instance of the problem and each language, we address the problem differently. We use an ensemble architecture to solve the Bot Detection for accounts that write in English and a single SVM for those who write in Spanish. For the Gender detection we use a single SVM architecture for both the languages, but we pre-process the tweets in a different way. Our final models achieve accuracy over the 90\% in the bot detection task, while for the gender detection, of 84.17\% and 77.61\% respectively for the English and Spanish languages.


# Getting Started

## How to install 

### Requirements

* git
* Python 3.7
* Pip

After pull the repository, you need to install all dependency. <br>
We suggest the use of python environment. <br>

```shell script
pip3 install -r --user requirements.txt
```
#### Install spacy
Install spacy globally with admin permission.
Execute the following command 
```shell script
python -m spacy download es_core_news_sm 
```
Enter in python3 shell and try to load the 'es_core_news_sm'
```python
import spacy
spacy.load('es_core_news_sm')
```


## Dataset Directory Structure
```
dataset
|
└───en
│   │   id1.xml
│   │   id2.xml
│   │   ...
|   |   ...
│   │   truth-train.txt
│   │   truth-dev.txt
│   
└───es
    │   id1.xml
    │   id2.xml
    │   ...
    |   ...
    │   truth-train.txt
    │   truth-dev.txt
```





# Authors

* **Andrea Bacciu**  - [github](https://github.com/andreabac3)
* **Massimo La Morgia**  - [github](https://github.com/andreabac3)
* **Eugenio Nerio Nemmi**  - [github](https://github.com/andreabac3)
* **Valerio Neri**  - [github](https://github.com/andreabac3)
