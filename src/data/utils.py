"""Data conversion helpers for nested named-entity recognition."""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _data_examples(text):
    return [block for block in text.replace("\r\n", "\n").strip().split("\n\n") if block]


def add_bos_eos(labels_file):
    """Add CoDeLin boundary rows to each sentence in a MaChAmp label file.

    The operation is idempotent so that evaluating existing predictions twice
    does not add duplicate markers.
    """
    path = Path(labels_file)
    examples = _data_examples(path.read_text(encoding="utf-8"))
    if not examples:
        raise ValueError(f"Label file is empty: {path}")

    output = []
    for example in examples:
        lines = example.splitlines()
        if lines[0].startswith("-BOS-") and lines[-1].startswith("-EOS-"):
            output.append("\n".join(lines))
            continue
        n_columns = len(lines[0].split("\t"))
        # Single-task MaChAmp output omits the token column expected by CoDeLin.
        n_columns = max(n_columns, 2)
        bos = "\t".join(["-BOS-"] * n_columns)
        eos = "\t".join(["-EOS-"] * n_columns)
        output.append("\n".join([bos, *lines, eos]))

    path.write_text("\n\n".join(output) + "\n", encoding="utf-8")
    return str(path)


def remove_comments(file):
    """Remove comment rows from a label file in place."""
    path = Path(file)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text(
        "".join(line for line in lines if not line.startswith("#")),
        encoding="utf-8",
    )


def parse_input(input_text):
    """Parse the four-column BIO format used by early preprocessing code."""
    parsed_data = []
    for example in _data_examples(input_text):
        lines = example.splitlines()
        sentence_info = lines[0]
        tokens = []
        entities = []
        current_entity = None
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            token_id, token, entity_tag, _nested_entity_tag = parts
            tokens.append(token)
            token_index = int(token_id) - 1
            if entity_tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = [entity_tag[2:], token_index, token_index]
            elif entity_tag.startswith("I-") and current_entity:
                current_entity[2] = token_index
            elif entity_tag == "O" and current_entity:
                entities.append(current_entity)
                current_entity = None
        if current_entity:
            entities.append(current_entity)
        parsed_data.append((sentence_info, tokens, entities))
    return parsed_data


def remove_features(file):
    """Keep the first and third columns of a three-column label file."""
    path = Path(file)
    output = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            output.append("")
            continue
        columns = line.split("\t")
        if len(columns) < 3:
            raise ValueError(f"Expected at least 3 columns at {path}:{line_number}")
        output.append("\t".join((columns[0], columns[2])))
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def format_output(parsed_data):
    formatted_output = []
    for _sentence_info, tokens, entities in parsed_data:
        ranges = [f"{start},{end} {entity_type}" for entity_type, start, end in entities]
        formatted_output.append(f"{' '.join(tokens)}\n{'|'.join(ranges)}")
    return "\n\n".join(formatted_output)


def parse_entities(entities_str):
    """Parse ``start,end LABEL`` annotations with inclusive token offsets."""
    entities = []
    for raw_entity in entities_str.split("|"):
        raw_entity = raw_entity.strip()
        if not raw_entity:
            continue
        try:
            span, entity_type = raw_entity.split(maxsplit=1)
            start, end = (int(value) for value in span.split(",", 1))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid entity annotation: {raw_entity!r}") from error
        if start < 0 or end < start:
            raise ValueError(f"Invalid inclusive span: {start},{end}")
        entities.append((start, end, entity_type))
    return sorted(entities, key=lambda entity: (entity[0], -entity[1], entity[2]))


def _validate_nested_entities(entities, token_count):
    for start, end, _label in entities:
        if end >= token_count:
            raise ValueError(f"Entity span {start},{end} exceeds {token_count} tokens")
    for index, first in enumerate(entities):
        for second in entities[index + 1 :]:
            first_start, first_end = first[:2]
            second_start, second_end = second[:2]
            if first_start < second_start <= first_end < second_end:
                raise ValueError(f"Crossing entity spans are not supported: {first}, {second}")
            if second_start < first_start <= second_end < first_end:
                raise ValueError(f"Crossing entity spans are not supported: {first}, {second}")


def build_tree(text_str, entities):
    """Build a parenthesized tree for nested, non-crossing entity spans."""
    words = text_str.split()
    entities = sorted(list(entities), key=lambda entity: (entity[0], -entity[1], entity[2]))
    _validate_nested_entities(entities, len(words))

    openings = {}
    for entity in entities:
        openings.setdefault(entity[0], []).append(entity)

    result = ["(ROOT"]
    stack = []
    for token_index, word in enumerate(words):
        for entity in openings.get(token_index, []):
            result.append(f" ({entity[2]}")
            stack.append(entity)
        result.append(f" {word}")
        while stack and stack[-1][1] == token_index:
            result.append(")")
            stack.pop()

    if stack:
        raise ValueError(f"Unclosed entity spans: {stack}")
    result.append(")")
    return "".join(result)


def nner_to_tree(text_str, entities_str):
    """Convert one NNER example to a parenthesized tree."""
    return build_tree(text_str, parse_entities(entities_str))


def iter_data_examples(text):
    """Yield ``(tokens, entity_set)`` pairs from the two-line data format."""
    for example_index, example in enumerate(_data_examples(text), 1):
        lines = example.splitlines()
        if len(lines) > 2:
            raise ValueError(f"Example {example_index} has more than two lines")
        tokens = lines[0].split()
        entities = {
            (label, start, end)
            for start, end, label in parse_entities(lines[1] if len(lines) == 2 else "")
        }
        _validate_nested_entities(
            [(start, end, label) for label, start, end in entities],
            len(tokens),
        )
        yield tokens, entities


def to_parenthesized(input_file_path, output_file_path):
    """Convert a complete NNER data file to one tree per line."""
    input_path = Path(input_file_path)
    trees = []
    for tokens, entities in iter_data_examples(input_path.read_text(encoding="utf-8")):
        text = " ".join(tokens).replace("(", "-LB-").replace(")", "-RB-")
        spans = [(start, end, label) for label, start, end in entities]
        trees.append(build_tree(text, spans))
    Path(output_file_path).write_text("\n".join(trees) + "\n", encoding="utf-8")
    return str(output_file_path)


def remove_bos_eos(input_file):
    """Remove CoDeLin boundary rows from a label file in place."""
    path = Path(input_file)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text(
        "".join(line for line in lines if "-BOS-" not in line and "-EOS-" not in line),
        encoding="utf-8",
    )


def _codelin_command(mode, encoding, input_file, output_file, multitask, codelin_dir):
    script = Path(codelin_dir) / "main.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"CoDeLin entry point not found at {script}. "
            "Run 'git submodule update --init --recursive'."
        )
    command = [
        sys.executable,
        str(script),
        "CONST",
        mode,
        encoding,
        str(input_file),
        str(output_file),
        "--sep",
        "[_]",
        "--ujoiner",
        "[+]",
        "--b_marker",
        "[b]",
    ]
    if multitask:
        command.extend(("--multitask", "--n_label_cols", "3"))
    if mode == "ENC":
        command.append("--ignore_postags")
    return command


def encode(encoding, trees_file, labels_file, multitask=True, codelin_dir=None):
    """Encode trees as sequence labels with CoDeLin."""
    codelin_dir = codelin_dir or PROJECT_ROOT / "CoDeLin"
    subprocess.run(
        _codelin_command(
            "ENC", encoding, trees_file, labels_file, multitask, codelin_dir
        ),
        check=True,
        cwd=PROJECT_ROOT,
    )
    return str(labels_file)


def decode(encoding, labels_file, trees_file, multitask=True, codelin_dir=None):
    """Decode sequence labels to trees with CoDeLin."""
    codelin_dir = codelin_dir or PROJECT_ROOT / "CoDeLin"
    subprocess.run(
        _codelin_command(
            "DEC", encoding, labels_file, trees_file, multitask, codelin_dir
        ),
        check=True,
        cwd=PROJECT_ROOT,
    )
    return str(trees_file)


def extract_entities_from_tree(tree):
    """Extract plain text and inclusive entity spans from an NLTK tree."""
    from nltk.tree import Tree

    def traverse_tree(subtree, position):
        text = []
        entities = []
        for node in subtree:
            if isinstance(node, Tree):
                entity_text, child_entities = traverse_tree(node, position)
                entity_length = len(entity_text.split())
                if node.label() != "ROOT":
                    entities.append(
                        (position, position + entity_length - 1, node.label())
                    )
                text.append(entity_text)
                entities.extend(child_entities)
                position += entity_length
            else:
                text.append(node)
                position += 1
        return " ".join(text), entities

    return traverse_tree(tree, 0)


def extract_entities_from_str(entities_str):
    """Return ``(label, start, end)`` tuples from an annotation line."""
    return {
        (label, start, end) for start, end, label in parse_entities(entities_str)
    }


def find_entities(file_path):
    """Return one exact-entity set per sentence in an NNER data file."""
    text = Path(file_path).read_text(encoding="utf-8")
    return [entities for _tokens, entities in iter_data_examples(text)]


def data_to_jsonlines(data_file, jsonlines_file):
    """Convert NNER data to JSON Lines while retaining inclusive offsets."""
    records = []
    text = Path(data_file).read_text(encoding="utf-8")
    for tokens, entities in iter_data_examples(text):
        mentions = []
        for label, start, end in sorted(entities, key=lambda item: (item[1], item[2])):
            mentions.append(
                {
                    "entity_type": label,
                    "start": start,
                    "end": end,
                    "text": " ".join(tokens[start : end + 1]),
                }
            )
        records.append({"tokens": tokens, "entity_mentions": mentions})

    with Path(jsonlines_file).open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(jsonlines_file)


def trees_to_data(trees_file, output_file):
    """Convert one parenthesized tree per line to NNER data format."""
    from nltk.tree import Tree

    trees = [
        Tree.fromstring(line)
        for line in Path(trees_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    examples = []
    for tree in trees:
        text, entities = extract_entities_from_tree(tree)
        text = text.replace("-LB-", "(").replace("-RB-", ")")
        annotations = "|".join(
            f"{start},{end} {label}"
            for start, end, label in sorted(entities, key=lambda item: (item[0], item[1]))
        )
        examples.append(f"{text}\n{annotations}")
    Path(output_file).write_text("\n\n".join(examples) + "\n", encoding="utf-8")
    return str(output_file)
