import os
import json
import nltk


def add_bos_eos(labels_file):
    """
    Add -BOS- and -EOS- markers to the selected files to allow CoDeLin to decode
    MaChAmp files that do not use these markers.
    """
    with open(labels_file, 'r') as f:
        text = f.read()
    sentences = text.split('\n\n')
    n_columns = len(text.split('\n')[0].split('\t'))
    if n_columns == 1:
        n_columns = 2 # chapuza
    print('n_columns:', n_columns)

    sentences = [
        '-BOS-\t'*(n_columns-1)+'-BOS-\n' +
        sentence + '\n'+
        '-EOS-\t'*(n_columns-1)+'-EOS-\n' 
        for sentence in sentences if sentence != ''
    ]

    with open(labels_file, 'w') as f:
        f.writelines('\n'.join(sentences) + '\n')
    
    return labels_file

def remove_comments(file):
    """
    Remove the comments from the .labels files
    """
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    output_lines = [line for line in lines if not line.startswith('#')]

    with open(file, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)

def parse_input(input_text):
    sentences = input_text.strip().split('\n\n')
    parsed_data = []

    for sentence in sentences:
        lines = sentence.strip().split('\n')
        sentence_info = lines[0]
        tokens = []
        entities = []
        current_entity = None

        for line in lines[1:]:
            parts = line.split('\t')
            if len(parts) == 4:
                token_id, token, entity_tag, nested_entity_tag = parts
                tokens.append(token)
                
                if entity_tag.startswith('B-'):
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = [entity_tag[2:], int(token_id)-1, int(token_id)]
                elif entity_tag.startswith('I-') and current_entity:
                    current_entity[2] = int(token_id)
                elif entity_tag == 'O' and current_entity:
                    entities.append(current_entity)
                    current_entity = None
        if current_entity:
            entities.append(current_entity)

        parsed_data.append((sentence_info, tokens, entities))

    return parsed_data

def remove_features(file):
    """
    Remove the second column of a 3-column .labels file
    """
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(file, 'w', encoding='utf-8') as f:
        for line in lines:
            if line == '\n':
                f.write('\n')
                continue
            line = line.split('\t')
            line = [line[0], line[2]]
            f.write("\t".join(line))
            
def format_output(parsed_data):
    formatted_output = []
    
    for sentence_info, tokens, entities in parsed_data:
        sentence_text = ' '.join(tokens)
        entity_ranges = []
        
        for entity in entities:
            entity_type, start, end = entity
            entity_range = f'{start},{end} {entity_type}'
            entity_ranges.append(entity_range)
        
        formatted_output.append(f"{sentence_text}\n{'|'.join(entity_ranges)}")
    
    return '\n\n'.join(formatted_output)

def nner_to_tree(text_str, entities_str):
    tokens = text_str.split()
    entities = entities_str.split('|')
    starts = []
    ends = []

    if len(entities) > 1:
        types = [entity.split(" ")[1] for entity in entities]
        for entity in entities:
            start, end = [int(n) for n in entity.split(" ")[0].split(',')]
            starts.append(start)
            ends.append(end)
    else:
        entities = []

    # Stack and output initialization
    stack = []
    output = ''

    # Start with the ROOT
    stack.append("ROOT")
    output += f'({stack[-1]}'

    # Processing tokens with stack
    for i, token in enumerate(tokens):
        # Handle opening new entities
        while i in starts:
            index = starts.index(i)
            entity_type = types[index]
            stack.append(entity_type)
            output += f' ({stack[-1]}'
            starts[index] = -1  # Mark this start as processed

        # Add the current token to the output
        output += f' {token}'

        # Handle closing entities
        while i in ends:
            index = ends.index(i)
            output += ')'
            stack.pop()  # Close the last opened entity
            ends[index] = -1  # Mark this end as processed

    # Finalize the output by closing the ROOT
    output += ')'

    return output

def to_parenthesized(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip().replace('\n\n\n', '\n\n').split('\n\n')

    trees = []

    for example in content:
        text_str = example.split('\n')[0].replace('(', '-LB-').replace(')', '-RB-')
        entities_str = example.split('\n')[1] if len(example.split('\n')) > 1 else ''
        
        # Parse entities
        entities = parse_entities(entities_str)

        # Create tree with nesting handled
        tree = build_tree(text_str, entities)
        trees.append(tree)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(trees))

    return output_file_path

def parse_entities(entities_str):
    """
    Parse the entity string into a list of tuples (start, end, entity_type).
    Example input: '0,5 ORG|4,5 GPE|12,12 GPE|14,15 ORG|14,19 GPE|22,22 FAC|22,29 GPE'
    """
    entities = []
    for entity in entities_str.split('|'):
        if entity:
            span, entity_type = entity.split()
            start, end = map(int, span.split(','))
            entities.append((start, end, entity_type))
    
    # Sort by start, then end (longest entities come first if they overlap)
    entities.sort(key=lambda x: (x[0], -x[1]))
    return entities


def build_tree(text_str, entities):
    """
    Build a nested tree string from the text and entities, handling nesting properly.
    """
    words = text_str.split()
    result = []
    stack = []  # Each element is (end_index, entity_type)

    for i, word in enumerate(words):
        # Close any entities that have ended before the current word
        indices_to_close = []
        for idx in reversed(range(len(stack))):
            if stack[idx][0] < i:
                _, entity_type = stack.pop(idx)
                result.append(")")
        
        # Open any new entities starting at current word
        while entities and entities[0][0] == i:
            start, end, entity_type = entities.pop(0)
            result.append(f"({entity_type}")
            stack.append((end, entity_type))

        # Add the word
        result.append(word)

        # Close any entities that end at current word
        indices_to_close = []
        for idx in reversed(range(len(stack))):
            if stack[idx][0] == i:
                _, entity_type = stack.pop(idx)
                result.append(")")

    # Close any remaining entities
    while stack:
        _, entity_type = stack.pop()
        result.append(")")

    return f"(ROOT {' '.join(result).replace(' )', ')')})"


def remove_bos_eos(input_file):
    """
    Remove the -BOS- and -EOS- rows from the .labels files
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    with open(input_file, 'w', encoding='utf-8') as file:
        for line in lines:
            if '-BOS-' not in line and '-EOS-' not in line:
                file.write(line)


def encode(encoding, trees_file, labels_file, multitask=True):
    """
    Encodes the input data into a .labels format using CoDeLin
    """
    if multitask:
        mt = '--multitask --n_label_cols 3'
    else:
        mt = ''
    codelin_script = f'python CoDeLin/main.py CONST ENC {encoding} {trees_file} {labels_file}  \
        --sep [_] --ujoiner [+] {mt} --ignore_postag' #separator not present in labels
    
    os.system(codelin_script)
    
    return labels_file

def decode(encoding, labels_file, trees_file, multitask=True):
    """
    Decodes the input data into a .trees format using CoDeLin
    """
    if multitask:
        mt = '--multitask --n_label_cols 3'
    else:
        mt = ''

    decoding_script =  f'python CoDeLin/main.py CONST DEC {encoding} {labels_file} {trees_file} \
        --sep [_] --ujoiner [+] {mt}'

    os.system(decoding_script)

    return trees_file

def extract_entities_from_tree(tree):
    """
    Given a tree, it extract the text and the entities
    """
    def traverse_tree(subtree, position):
        text = []
        entities = []
        
        for node in subtree:
            if isinstance(node, nltk.Tree):
                entity_type = node.label()
                entity_text, child_entities = traverse_tree(node, position)
                
                start_position = position
                entity_len = len(entity_text.split())
                end_position = start_position + entity_len - 1
                
                text.append(entity_text)
                if entity_type != 'ROOT':
                    entities.append((start_position, end_position, entity_type))
                
                position += entity_len
                entities.extend(child_entities)
            else:
                text.append(node)
                position += 1
        
        return " ".join(text), entities

    text, entities = traverse_tree(tree, 0)
    return text, entities

def extract_entities_from_str(entities_str):
    """
    Given the entity annotation in string format, returns a list of dicts 
    with label, start, and end of the entity.
    """
    entities = set([
        (
            entity.split(' ')[1],
            int(span.split(',')[0]),
            int(span.split(',')[1])
        )
        for entity in entities_str.split('|')
        for span in [entity.split(' ')[0]]
    ])
    
    return entities


def find_entities(file_path):
    """
    Given a NNER data file, it returns a list of lists of tuples with
    start, end and entity type for every sentence.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().replace('\n\n\n', '\n\n').split('\n\n')

    entities_str_list = [line.split('\n')[1] if len(line.split('\n')) == 2 else '' for line in lines]
    entities_list = []

    for entities_str in entities_str_list:
        if entities_str != '':
            entities = extract_entities_from_str(entities_str)
        else:
            entities = set()

        entities_list.append(entities)

    return entities_list

def data_to_jsonlines(data_file, jsonlines_file): # we repeat code to test fast.
    data_list = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.read().replace('\n\n\n', '\n\n').split('\n\n')

    for line in lines:
        tokens = line.split('\n')[0].split()
        entities_str = line.split('\n')[1] if len(line.split('\n')) == 2 else ''
        if tokens == []:
            continue

        data = {}

        data["tokens"] = tokens
        #data["doc_id"] = ""
        #data["sent_id"] = ""
        data["entity_mentions"] = []

        for entity in entities_str.split('|'):
            if len(entity.split(' ')) == 1:
                continue

            span = entity.split(' ')[0] 
            entity_type = entity.split(' ')[1]
            start = int(span.split(',')[0])
            end = int(span.split(',')[1])
            text = ' '.join(tokens[start:end])
            
            data["entity_mentions"].append({
                "entity_type": entity_type,
                "start": start,
                "end": end,
                "text": text
            })
        
        
        data_list.append(data)
    
    with open(jsonlines_file, 'w', encoding='utf-8') as json_file:
        for entry in data_list:
            json_file.write(json.dumps(entry) + '\n')

    return jsonlines_file


def trees_to_data(trees_file, output_file):
    """
    Given a .trees file, returns .data file 
    """
    output_data = ''
    with open(trees_file, 'r') as f:
        trees = [nltk.Tree.fromstring(sentence) for sentence in f.readlines()]
    
    for tree in trees:
        text, entities = extract_entities_from_tree(tree)
        text = text.replace('-LB-', '(').replace('-RB-', ')')
        entities = sorted(entities, key=lambda x: ((x[0], x[1])))
        entity_annotations = [f"{start},{end} {etype}" for start, end, etype in entities]
        annotations = "|".join(entity_annotations)
        output_data += f"{text}\n{annotations}\n\n"

    with open(output_file, 'w') as f:
        f.write(output_data)

    return output_file
