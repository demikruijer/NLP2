import Checklist
from Checklist.editor import Editor
from Checklist.perturb import Perturb



if __name__ == "__main__":
    editor = Editor()

    with open("data1/olid-subset-diagnostic-tests.csv", encoding='utf8') as dev_file:
        data_df = pd.read_csv(dev_file)
        data = train_data["text"]

    # approach 1
    data[0], Perturb.add_typos(data[0])

    # approach 2
    ret = Perturb.perturb(data, Perturb.add_typos, nsamples=1)
    ret.data