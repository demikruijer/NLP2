# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class
import pandas as pd


def clean_data():
    model_output = pd.read_csv("random_baseline.tsv", sep="\t", encoding='utf8')
    model_output.columns=["tweet", "label", "prediction"]
    model_output = model_output.dropna()
    print(model_output.head())
    model_output.to_csv("model_out_maj.csv")

if __name__ == '__main__':
    #Per class, per model: Precision, Recall, F1
    clean_data()
    model_output = pd.read_csv("model_out_maj.csv")

    Off = model_output[model_output["label"] == 1]
    Not = model_output[model_output["label"] == 0]

    SizeOff = len(Off)
    SizeNot = len(Not)


    #Precision = True Positives / (True Positives + False Positives)

    trueposOff = Off[Off["label"] == Off["prediction"]].shape[0]
    falseposOff =  Not[Not["prediction"] == 1].shape[0]

    trueposNot = Not[Not["label"] == Not["prediction"]].shape[0]
    falseposNot = Off[Off["prediction"] == 0].shape[0]

    if (trueposOff + falseposOff) == 0:
        precisionOff = 0
    else:
        precisionOff = trueposOff / (trueposOff + falseposOff)

    precisionNot = trueposNot / (trueposNot + falseposNot)

    #Recall = True Positives / (True Positives + False Negatives)
    falsenegOff = Off[Off["prediction"] == 0].shape[0]
    recallOff = trueposOff / (trueposOff + falsenegOff)

    falsenegNot = Not[Not["prediction"] == 1 ].shape[0]
    recallNot = trueposNot / (trueposNot + falsenegNot)

     #F1 = 2 * (Precision * Recall) / (Precision + Recall)
    if precisionOff == 0 or recallOff == 0:
        F1Off = 0
    elif precisionOff != 0 or recallOff != 0:
        F1Off = 2 * (precisionOff * recallOff) / (precisionOff + recallOff)

    if precisionNot == 0 or recallNot == 0:
        F1Not = 0
    elif precisionNot != 0 or recallNot != 0:
        F1Not = 2 * (precisionNot * recallNot) / (precisionNot + recallNot)

    macroavprecision = 0.5 * precisionOff + 0.5 * precisionNot
    macroavrecall = 0.5 * recallOff + 0.5 * recallNot
    macroavF1 = 0.5 * F1Off + 0.5 * F1Not

    totallen = SizeNot + SizeOff
    weightedavprecision = SizeOff/totallen * precisionOff + SizeNot/totallen * precisionNot
    weightedavrecall = SizeOff/totallen * recallOff + SizeNot/totallen * recallNot
    weigtedavF1 = SizeOff/totallen * F1Off + SizeNot/totallen * F1Not

    print("Precision Off = ", precisionOff,  "  Not = ", precisionNot, "Macro-average", macroavprecision, "Weighted average", weightedavprecision)
    print("Recall Off = ", recallOff, " Not = ", recallNot, "Macro-average", macroavrecall, "Weighted average", weightedavrecall)
    print("F1Off", F1Off, "F1Not", F1Not, "Macro-average", macroavF1, "Weighted average", weigtedavF1)


