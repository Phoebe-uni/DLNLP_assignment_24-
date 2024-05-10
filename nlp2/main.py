'''
Written by YYF.
'''
import argparse
from A.lstm import lstm_main as LSTM_Main
from A.AnalysisTSE import AnalysisText as AnalysisText
from B.roberta import roberta_main as Roberta_Main
from C.bert import bert_main  as Bert_Main

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(
        description='DNLP Assignment - Processing Twitter Sentiment Extraction')

    parser.add_argument('--task',
                        help="'A','B' or 'C', select A: Analysis TSE and use LSTM Model, B: use Roberta Model C: use Bert Model",
                        type=str,
                        required=True)

    args = parser.parse_args()
    taskid = args.task.lower()

    if taskid == 'a':
        AnalysisText()
        LSTM_Main()
    elif taskid == 'b':
        Roberta_Main()
    else:
        Bert_Main()


