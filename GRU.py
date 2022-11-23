import torch
from torch import nn
from torch.nn import functional
import random
import json
import os
import io
import re
from tensorflow.python.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tqdm import tqdm
from rouge import Rouge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS = 128930
EOS = 128931

random.seed(1801)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.model = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.model(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_prob=0.2, max_length=500):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.dropout_probability = dropout_prob

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.model = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_output):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = functional.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_combined = torch.bmm(attn_weights.unsqueeze(0), encoder_output.unsqueeze(0))

        output = torch.cat((embedded[0], attn_combined[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = functional.relu(output)
        output, hidden = self.model(output, hidden)

        output = functional.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(input_tensor, output, encoder, decoder, enc_optimizer, dec_optimizer, loss_criteria, teacher_prob=0.5):
    max_length = 500
    encoder_hidden = encoder.initHidden()

    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()

    input_length = len(input_tensor)
    output_length = len(output)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for idx in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[idx], encoder_hidden)
        encoder_outputs[idx] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS]], device=device)

    decoder_hidden = encoder_hidden

    use_tf = True if random.random() < teacher_prob else False

    if use_tf:
        for idx in range(output_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += loss_criteria(decoder_output, output[idx])
            decoder_input = output[idx]
    else:
        for idx in range(output_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += loss_criteria(decoder_output, output[idx])
            if decoder_input.item() == EOS:
                break

    loss.backward()

    enc_optimizer.step()
    dec_optimizer.step()

    return loss.item() / output_length


def trainer(x_train, y_train, encoder, decoder, epochs, lr=0.01):
    print("Starting Training...")
    max_length = 500
    prev_loss = None
    losses = []
    total_losses = 0

    enc_optimizer = torch.optim.SGD(encoder.parameters(), lr=lr)
    dec_optimizer = torch.optim.SGD(decoder.parameters(), lr=lr)

    loss_criteria = nn.NLLLoss()

    input_data = [inputs for inputs in zip(x_train, y_train)]
    # sample_size = int(0.1 * len(input_data))
    # print(f"Training Sample Size: {sample_size}\n")

    for epoch in range(1, epochs + 1):
        total_loss = 0
        #current_input = random.sample(input_data, sample_size)
        for x, y in tqdm(input_data, total=len(input_data)):
            input_length = x.size(0)
            if input_length > max_length:
                continue
            loss = train(x.view(-1, 1).to(device), y.view(-1, 1).to(device), encoder, decoder, enc_optimizer, dec_optimizer, loss_criteria)
            total_loss += loss
        total_losses /= len(x_train)
        losses.append(total_loss)
        print(f"Epoch {epoch}, Loss: {total_loss}")
        if prev_loss is None or total_loss < prev_loss:
            prev_loss = total_loss
            torch.save(encoder.state_dict(), "./GRU-model/encoder.pt")
            torch.save(decoder.state_dict(), "./GRU-model/decoder.pt")
            print("Loss Improved, Model Saved.\n")
        else:
            print()

def make_prediction(input_text, encoder, decoder):
    max_length = 500
    input_length = len(input_text)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    encoder_hidden = encoder.initHidden()
    for idx in range(input_length):
        encoder_output, encoder_hidden = encoder(input_text[idx], encoder_hidden)
        encoder_outputs[idx] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS]], device=device)

    decoder_hidden = encoder_hidden

    output_text = []

    for idx in range(100):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        if decoder_input.item() == EOS:
            break
        output_text.append(decoder_input.item())

    return output_text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\([^()]*\)', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def load_rawdata(fname):
    print("Loading Normal Text...")
    normal_text = []
    with open(os.path.join(fname, "normal.aligned")) as f:
        lines = f.readlines()
        current = []
        topic = None
        for line in lines:
            data = line.split("\t")
            normal_text.append(clean_text(data[2]))
            # if data[0] == topic:
            #     current.append(data[2])
            # else:
            #     topic = data[0]
            #     if len(current) != 0:
            #         normal_text.append(clean_text(' '.join(x for x in current)))
            #     current = []

    print("Loading Simplified Text...")
    simple_text = []
    with open(os.path.join(fname, "simple.aligned")) as f:
        lines = f.readlines()
        current = []
        topic = None
        for line in lines:
            data = line.split("\t")
            simple_text.append(clean_text(data[2]))
            # if data[0] == topic:
            #     current.append(data[2])
            # else:
            #     topic = data[0]
            #     if len(current) != 0:
            #         simple_text.append(clean_text(' '.join(x for x in current)))
            #     current = []

    return normal_text, simple_text


X, Y = load_rawdata("./datasets/wiki-dataset")
input_data = [(x, y) for x, y in zip(X, Y)]
current_input = random.sample(input_data, 20)
X = [x for x, y in current_input]
Y = [y for x, y in current_input]

X_extra, Y_extra = load_rawdata("./datasets/")

if not os.path.exists("./GRU-model"):
    os.makedirs("./GRU-model")

if os.path.exists("./GRU-model/tokenizer.json"):
    with open('./GRU-model/tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
else:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X + Y)
    tokenizer_json = tokenizer.to_json()
    with io.open('./GRU-model/tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))



X_train = [torch.tensor([SOS] + x + [EOS]) for x in tokenizer.texts_to_sequences(X)]
Y_train = [torch.tensor([SOS] + y + [EOS]) for y in tokenizer.texts_to_sequences(Y)]

print(f"Number of Datapoints: {len(X_train)}")

hidden_size = 512
encoder = Encoder(128932, hidden_size).to(device)
attn_decoder = AttnDecoder(hidden_size, 128932).to(device)

# encoder.load_state_dict(torch.load("./GRU-model/encoder.pt"))
# attn_decoder.load_state_dict(torch.load("./GRU-model/decoder.pt"))

trainer(X_train, Y_train, encoder, attn_decoder, 1000)

encoder.load_state_dict(torch.load("./GRU-model/encoder.pt"))
attn_decoder.load_state_dict(torch.load("./GRU-model/decoder.pt"))

rouge = Rouge()
generated = []
for text, correct in zip(X_extra, Y_extra):
    input_text = torch.tensor([SOS] + tokenizer.texts_to_sequences([text])[0] + [EOS])
    ans = tokenizer.sequences_to_texts([make_prediction(input_text.to(device), encoder, attn_decoder)])
    test_tensor = [input_text[1:-1].tolist()]
    input_text = tokenizer.sequences_to_texts(test_tensor)
    print("\n")
    print(f"Original Text: {input_text[0]}")
    print(f"Simplified Text: {ans[0]}")
    print(f"Simplified Text: {correct}")
    generated.append(ans[0])

print()

scores = rouge.get_scores(generated, Y_extra)
avg_scores = {'rouge-1': {'f':0, 'p':0, 'r':0}, 'rouge-2': {'f':0, 'p':0, 'r':0}, 'rouge-l': {'f':0, 'p':0, 'r':0}}
for score in scores:
    for k, v in score.items():
        for k2, v2 in v.items():
            avg_scores[k][k2] += v2

for k, v in avg_scores.items():
    for k2, v2 in v.items():
        avg_scores[k][k2] = v2 / len(scores)

print(json.dumps(avg_scores, indent=2))