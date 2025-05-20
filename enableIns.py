moreCriteria = '''
Attention:\n1.Generate triples as many as possible.
'''

graphusionCriteria = '''
Given a document segments, and a query keyword, do the following:
1. Extract the query concept and some in-domain concepts from the document segments, these concepts should be fine-grained: could be introduced by a lecture slide page, or a whole lecture, or possibly to have a Wikipedia page.
2. Determine the relationships between the query concept and the extracted concepts from Step 1, in a triplet format: <'head concept', 'relation', 'tail concept'>. The relationship should be functional, aiding learners in understanding the knowledge. The query concept can be the head concept or tail concept. We define 7 types of the relations:
    a) Compare: Represents a relationship between two or more entities where a comparison is being made. For example, 'A is larger than B' or 'X is more efficient than Y.'
    b) Part-of: Denotes a relationship where one entity is a constituent or component of another. For instance, 'Wheel is a part of a Car.'
    c) Conjunction: Indicates a logical or semantic relationship where two or more entities are connected to form a group or composite idea. For example, 'Salt and Pepper.'
    d) Evaluate-for: Represents an evaluative relationship where one entity is assessed in the context of another. For example, 'A tool is evaluated for its effectiveness.'
    e) Is-a-Prerequisite-of: This dual-purpose relationship implies that one entity is either a characteristic of another or a required precursor for another. For instance, 'The ability to code is a prerequisite of software development.'
    f) Used-for: Denotes a functional relationship where one entity is utilized in accomplishing or facilitating the other. For example, 'A hammer is used for driving nails.'
    g) Hyponym-Of: Establishes a hierarchical relationship where one entity is a more specific version or subtype of another. For instance, 'A Sedan is a hyponym of a Car.'
3. Some relation types are strictly directional. For example, 'A tool is evaluated for B' indicates <'A', 'Evaluate-for', 'B'>, NOT <'B', 'Evaluate-for', 'A'>. Among the seven relation types, only 'a) Compare' and 'c) Conjunction' are not direction-sensitive.
4. You can also extract triplets from the extracted concepts, and the query concept may not be necessary in the triplets.
5. Your answer should ONLY contain a list of triplets, each triplet is in this format: <'concept', 'relation', 'concept'>. For example: <'concept', 'relation', 'concept'>, <'concept', 'relation', 'concept'>. No numbering and other explanations are needed.
6. If document segments is empty, output None.
'''

graphjudgerCriteria = '''
Goal:\nTransform the text into a semantic graph (a list of triples) with the given document segments and keywords. 
In other words, You need to find relations between the given keywords with the given document segments.\n
Attention:\n1.Generate triples as many as possible.
2.Make sure each item in the list is a triple with strictly three items.\n\n
Here are two examples:\n
Example#1: \ndocument segments: Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust.\n
Keywords List: ['Shotgate Thickets', 'Nature reserve', 'United Kingdom', 'Essex Wildlife Trust']\n
Semantic Graph: <'Shotgate Thickets', 'instance of', 'Nature reserve'>, <'Shotgate Thickets', 'country', 'United Kingdom'>, <'Shotgate Thickets', 'operator', 'Essex Wildlife Trust'>. \n 
Example#2:\ndocument segments: The Eiffel Tower, located in Paris, France, is a famous landmark and a popular tourist attraction.\n
It was designed by the engineer Gustave Eiffel and completed in 1889.\n
Keywords List: ['Eiffel Tower', 'Paris', 'France', 'landmark', 'Gustave Eiffel', '1889']\n
Semantic Graph: <'Eiffel Tower', 'located in', 'Paris'>, <'Eiffel Tower', 'located in', 'France'>, <'Eiffel Tower', 'instance of', 'landmark'>, <'Eiffel Tower', 'attraction type', 'tourist attraction'>, <'Eiffel Tower', 'designed by', 'Gustave Eiffel'>, <'Eiffel Tower', 'completion year', '1889'>\n
'''

graphjudger_keyword_extraction = '''
Goal:\nExtract a list of important keywords from the following text.\n
Example 1:
Text: Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust.
Keywords: ['Shotgate Thickets', 'Nature reserve', 'United Kingdom', 'Essex Wildlife Trust']\n
Example 2:
Text: Garczynski Nunatak is a cone-shaped nunatak near Mount Brecher in Antarctica.
Keywords: ['Garczynski Nunatak', 'nunatak', 'Mount Brecher', 'Antarctica']\n
'''

graphjudger_chunks_denoising = '''
Goal:\nDenoise the raw text with the given keywords, which means remove the unrelated text and make it more formatted.\n\n
Here are two examples:\n
Example 1:\nRaw text: 'Zakria Rezai (born 29 July 1989) is an Afghan footballer who plays for Ordu Kabul F.C., which is a football club from Afghanistan. He is also an Afghanistan national football team player, and he has 9 caps in the history. He wears number 14 on his jersey and his position on field is centre back.'\n
Keywords: ['Zakria Rezai','footballer','Ordu Kabul F.C.','Afghanistan','29 July 1989']\n
Denoised text: 'Zakria Rezai is a footballer. Zakria Rezai is a member of the sports team Ordu Kabul F.C. Zakria Rezai has the citizenship of Afghanistan. Zakria Rezai was born on July 29, 1989. Ordu Kabul F.C. is a football club. Ordu Kabul F.C. is based in Afghanistan.'\n\n
Example 2:\nRaw text: 'Elizabeth Smith, a renowned British artist, was born on 12 May 1978 in London. She is specialized in watercolor paintings and has exhibited her works in various galleries across the United Kingdom. Her most famous work, 'The Summer Breeze,' was sold at a prestigious auction for a record price. Smith is also a member of the Royal Society of Arts and has received several awards for her contributions to the art world.'\n
Keywords: ['Elizabeth Smith', 'British artist', '12 May 1978', 'London', 'watercolor paintings', 'United Kingdom', 'The Summer Breeze', 'Royal Society of Arts']\n
Denoised text: 'Elizabeth Smith is a British artist. Elizabeth Smith was born on May 12, 1978. Elizabeth Smith was born in London. Elizabeth Smith specializes in watercolor paintings. Elizabeth Smith's artwork has been exhibited in the United Kingdom. 'The Summer Breeze' is a famous work by Elizabeth Smith. Elizabeth Smith is a member of the Royal Society of Arts.'\n\n
'''

llmkg_conflict_resolution_prompt = """
You are a strict verifier of knowledge graph triples. 
You need to do two things based only on the given triples and document segments:

1. Identify conflicting triples:
- Carefully examine the provided triples.
- Only consider triples as conflicting if the conflict is explicit and unambiguous.
- If you are not absolutely certain that two triples conflict, assume that they do not conflict.

2. Decide based on document evidence which conflicting triples must be removed:
- Use the document segments to judge whether any conflicting triples must be removed to resolve the conflict.
- Only remove triples if absolutely necessary, with clear evidence.
- If document evidence is unclear, do NOT suggest removal.

If no conflicting triples are detected, you must respond exactly with "No conflicting triples".
If no triples need to be removed, you must respond exactly with "No triples to remove".
If any conflicting triples are detected, or if any triples need to be removed, list them strictly in the format: <'head', 'relation', 'tail'>. And your response must clearly separate the two sections — Conflicting Triples and Removal Triples — following the exact format shown below.

## Example 1
[Input] 
Triples to Examine: <'disease', 'causes', 'fever'>, <'fever', 'causes', 'disease'>
Document Segments: "Fever is a common symptom that occurs as a result of various diseases. Diseases such as infections or inflammatory conditions often lead to an elevation in body temperature. Fever itself is not typically a cause of disease but rather a response to an underlying condition."

[Output] 
Conflicting Triples: <'disease', 'causes', 'fever'>, <'fever', 'causes', 'disease'>
Removal Triples: <'fever', 'causes', 'disease'>
Explanation: They claim opposite causal directions between the same entities, thus conflict detected. Only one direction can be true, and the conflict must be resolved by removing one.

## Example 2
[Input]
Triples to Examine: <'virus', 'affects', 'immune system'>, <'immune system', 'responds to', 'virus'>
Document Segments: "When a virus enters the human body, it affects the immune system by triggering a defense response. The immune system, in turn, detects the virus and activates immune pathways to neutralize the threat."

[Output] 
Conflicting Triples: No conflicting triples
Removal Triples: No triples to remove
Explanation: These two statements describe different perspectives, not a contradiction. No conflict.

## Example 3  
[Input]
Triples to Examine: <'Vitamin D supplementation', 'reduces risk of', 'respiratory infections'>, <'Vitamin D supplementation', 'has no significant effect on', 'respiratory infections'>
Document Segments: "Several randomized controlled trials have found that vitamin D supplementation reduces the risk of respiratory infections, particularly in individuals with baseline vitamin D deficiency. However, other studies have reported no significant overall benefit, especially in populations without deficiency." 

[Output]
Conflicting Triples: <'Vitamin D supplementation', 'reduces risk of', 'respiratory infections'>, <'Vitamin D supplementation', 'has no significant effect on', 'respiratory infections'>
Removal Triples: No triples to remove
Explanation: Although the two triples describe different conclusions, the document acknowledges variability across studies without directly refuting either statement. Thus, no explicit conflict is confirmed.

Here are the triples to examine:
{triples_str}

And the relevant document segments:
{chunks_str}
"""

graphusion_conflict_resolution_prompt = """
###Instruction: You are a knowledge graph builder. Now please fuse the following triples into a single graph by detecting the conflicting triples and remove the unnecessary triples.
Here are the triples to examine:
{triples_str}

Rules for Fusing the Graphs:
1. Union the concepts and edges.
2. If two concepts are similar, or they are referring to the same concept, combine them as one concept by keeping the meaningful or specific one. For example, "lstm" versus "long short-term memory", please keep "long short-term memory".
3. We only allow one relation to exist between two concepts, if there is a conflict, read the following "##background" to help you keep the correct one. For example, <'ROUGE', 'Evaluate-for', 'question answering model'> and <'ROUGE', 'Used-for', 'question answering model'> are considered to be conflicts.
4. Once step 3 is done, consider every possible concept pair, which did not occur in step 2. For example, take a concept in G1, and match a concept from G2. And look at the "##background", and summarize new triplets.

Hint: the relation types and their definition. You can use it to do Step 3:
   a) Compare: Represents a relationship between two or more entities where a comparison is being made. For example, "A is larger than B" or "X is more efficient than Y."
   b) Part-of: Denotes a relationship where one entity is a constituent or component of another. For instance, "Wheel is a part of a Car."
   c) Conjunction: Indicates a logical or semantic relationship where two or more entities are connected to form a group or composite idea. For example, "Salt and Pepper."
   d) Evaluate-for: Represents an evaluative relationship where one entity is assessed in the context of another. For example, "A tool is evaluated for its effectiveness."
   e) Is-a-Prerequisite-of: This dual-purpose relationship implies that one entity is either a characteristic of another or a required precursor for another. For instance, "The ability to code is a prerequisite of software development."
   f) Used-for: Denotes a functional relationship where one entity is utilized in accomplishing or facilitating the other. For example, "A hammer is used for driving nails."
   g) Hyponym-Of: Establishes a hierarchical relationship where one entity is a more specific version or subtype of another. For instance, "A Sedan is a hyponym of a Car."

##Background: {chunks_str}

###Output Instruction: 
(1) If no conflicting triples are detected, you must respond exactly with "No conflicting triples".
(2) If no triples need to be removed, you must respond exactly with "No triples to remove".
(3) If any conflicting triples are detected, or if any triples need to be removed, list them strictly in the format: <'head', 'relation', 'tail'>. And your response must clearly separate the two sections — Conflicting Triples and Removal Triples. No other explanations or numbering are needed. Only triplets, no intermediate results. 
"""