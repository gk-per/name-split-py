# name-split-py
python distilbert

Take an input name, say "Chris Stansbury", or "America Ferrera", or "Jean-Louis Dumas", or, "Serena Van der Woodsen" and split them into {First name, Last name} pairs, e.g.: {Chris, Stansbury}, {America, Ferrera}, {Jean-Louis, Dumas}, {Serena, Van der Woodsen}.

The attempted implementation is to use the base DistilBERT model with the token classification architecture and training data to train additional NER labels "First name" and "Last name". 