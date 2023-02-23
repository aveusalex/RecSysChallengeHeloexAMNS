import torch
from transformers import BertTokenizer, BertModel
import time


# Load pretrained model/tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to('cuda')


#%%
def get_bert_embedding(text_list):
    # if text_list is over than 15 items, split it in groups of 15
    # print(len(text_list))
    if len(text_list) > 15:
        # print("Text list is over than 15 items, split it in groups of 15")
        text_list = [text_list[i:i + 15] for i in range(0, len(text_list), 15)]
    else:
        text_list = [text_list]
    embs = []
    for text in text_list:
        start = time.time()
        tokens = []
        # clip text if it is too long (more than 512 tokens)
        for idx in range(len(text)):
            if len(text[idx]) > 512:
                text[idx] = text[idx][:510]
            # Add the special tokens.
            marked_text = "[CLS] " + text[idx] + " [SEP]"
            # Split the sentence into tokens.
            tokenized_text = tokenizer.tokenize(marked_text)
            # padding if text is less than 512 tokens
            if len(tokenized_text) < 512:
                tokenized_text = tokenized_text + ["[PAD]"] * (512 - len(tokenized_text))
            # Map the token strings to their vocabulary indexes.
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens.append(indexed_tokens)
        # Convert inputs to PyTorch tensors and put them on GPU
        tokens_tensor = torch.tensor(tokens).to('cuda')

        # Put the model in "evaluation" mode,meaning feed-forward operation.
        model.eval()
        # Run the text through BERT, and collect all the hidden states produced from all 12 layers.
        with torch.no_grad():
            outputs = model(tokens_tensor)[2][-4:]

        # sum of last four layer
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.sum(1)
        # mean of the tokens, results in one vector of 768 dimensions per text
        outputs = torch.mean(outputs, 1).squeeze(0).cpu().numpy()
        print("Time to get embedding: ", time.time() - start)
        embs.append(outputs)
    return embs


if __name__ == '__main__':
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    text = "THE PIZZA WAS AMAZING!"
    text2 = "I've never had a pizza like this before!"
    text3 = "The pizza was really good!"
    text4 = "Really liked the pizza."
    text5 = "Really bad taste. Awful pizza."
    text6 = "I was really disappointed with the pizza."
    embs = get_bert_embedding([text, text2, text3, text4, text5, text6])
    print(embs)
    # usando t sne para reduzir a dimensionalidade e plotar os embeddings
    # tsne = TSNE(n_components=2, random_state=0)
    # embs = tsne.fit_transform(embs)
    # plt.figure(figsize=(10, 9))
    # plt.scatter(x=embs[0, 0], y=embs[0, 1])
    # plt.text(x=embs[0, 0], y=embs[0, 1], s=text)
    # plt.scatter(x=embs[1, 0], y=embs[1, 1])
    # plt.text(x=embs[1, 0], y=embs[1, 1], s=text2)
    # plt.scatter(x=embs[2, 0], y=embs[2, 1])
    # plt.text(x=embs[2, 0], y=embs[2, 1], s=text3)
    # plt.scatter(x=embs[3, 0], y=embs[3, 1])
    # plt.text(x=embs[3, 0], y=embs[3, 1], s=text4)
    # plt.scatter(x=embs[4, 0], y=embs[4, 1])
    # plt.text(x=embs[4, 0], y=embs[4, 1], s=text5)
    # plt.scatter(x=embs[5, 0], y=embs[5, 1])
    # plt.text(x=embs[5, 0], y=embs[5, 1], s=text6)
    #
    # plt.legend()
    # plt.show()
