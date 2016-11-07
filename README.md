# cheml

Set of tools for machine learning with molecules.

## Export a tar.bz2 file with the source

git archive master --output=cheml.zip

or for a certain commit

git archive commit-id --output=cheml.zip

where commit-id is the id of the commit,
for instance  a27e4466cec42b84e937ac7574884bf2ffedbba5

for a bzip2:
git archive master | bzip2 >source-tree.tar.bz2
