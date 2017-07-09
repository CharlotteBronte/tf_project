# nnlm-rnn实现的语言模型
## Requirements
## Usage
* python(above 2.7)    script/lstm_nnlm.py    cofigs/nnlm_test.ini

## Custom

## To Do List

## Updata

## Architecture
* script: all runnable script| 包含了所有的可执行脚本
* configs: all config file with .ini format(include model parameters) | 所有模型读取所需的配置文件
* logs: all logs(you'd better config your log with child directory)  | 所有日志文件（最好以子目录指定）
* model: all nnlm model, new if not exist(you'd better config your log with child directory) | 所有nnlm的模型
* data: test and train data for nnlm model with chinese segmented qa | 所有的测试和训练数据（所有数据均为qa对并且是使用jieba分词的）

## Preference
[NNLM](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf){:target="_blank"}
[RNNLM Toolkit](http://www.fit.vutbr.cz/~imikolov/rnnlm/){:target="_blank"}
