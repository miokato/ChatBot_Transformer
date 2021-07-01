from app.train import Trainer, Preprocessor, Symbol

dir_path = 'data/narou/parsed'
symbol = Symbol()
p = Preprocessor(symbol)
p.create_vocab(dir_path)
trainer = Trainer(p.vocab, symbol, p.text_transforms, p.collate_fn)
trainer.load('chatbot.pth')
res = trainer.predict('おはよう')
print(res)