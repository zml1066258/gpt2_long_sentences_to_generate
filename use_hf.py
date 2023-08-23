# from transformers import pipeline, set_seed
# generator = pipeline('text-generation', model='gpt2')
# set_seed(42)
# generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
# print(generator)



from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator("The White man worked as a", max_length=10, num_return_sequences=5)
print(generator)