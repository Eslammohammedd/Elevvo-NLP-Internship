# Models Directory

This directory is intended to store any custom trained NER models.

For Task 4, we primarily rely on pre-trained `spaCy` models:
- `en_core_web_sm` (Small, efficient model for basic NER)
- `en_core_web_md` (Medium, higher accuracy model using word vectors)

Since these are downloaded dynamically via `spacy.cli.download()`, they aren't stored as standalone files here. However, if we were to fine-tune a model using the `CoNLL003` dataset in the future, the `.spacy` exported model binaries would be saved in this directory.
