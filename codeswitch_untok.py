import sys
argv = sys.argv
def parse_dict(path_to_dict):
    with open(path_to_dict) as file:
        word_dictionary = {}
        for line in file.readlines():
            striped = line.strip().lower()
            columns = striped.split('\t')
            try:
                word_dictionary[columns[1]] = columns[0]
            except:
                print(columns)
                print(line)
        return word_dictionary

dictionary = parse_dict(argv[3])

sentence = [] 
with open(argv[1], 'r') as f:
    sentence = [line.strip() for line in list(f.readlines())]
code_switch_pairs = []
counter = 0
for sent in sentence:
    splitted = sent.split(' ')
    change = 0
    for word in splitted:
        if word in dictionary:
            change += 1
    score = change/len(splitted)
    if score > 0:
        counter += 1    
    translated = [dictionary.get(word, word) for word in splitted]
    trans_string = str.join(' ', translated)
    code_switch_pairs.append(trans_string)
with open(argv[2], 'w') as f:
    for sent in code_switch_pairs:
        f.write(sent + "\n")

print(counter)