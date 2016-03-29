import sys
start = { 'topic': '<topic>', 'number': '<number>', 'title': '<title>', 'question': '<question>', 'narrative': '<narrative>', 'concepts': '<concepts>' }
close = { 'topic': '</topic>', 'number': '</number>', 'title': '</title>', 'question': '</question>', 'narrative': '</narrative>', 'concepts': '</concepts>' }
with open(sys.argv[1]) as fp:
    content = ''.join([x.strip() for x in fp.readlines()])
    for unparsed_query in content.split(start['topic'])[1:]:
        number = unparsed_query.split(start['number'])[1].split(close['number'])[0][-3:]
        title = unparsed_query.split(start['title'])[1].split(close['title'])[0]
        question = unparsed_query.split(start['question'])[1].split(close['question'])[0][2:]
        narrative = unparsed_query.split(start['narrative'])[1].split(close['narrative'])[0][6:]
        concepts = unparsed_query.split(start['concepts'])[1].split(close['concepts'])[0]
        
        print(number)
        print(title)
        print(question)
        print(narrative)
        print(concepts)

        print()
