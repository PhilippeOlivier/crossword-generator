"""Generate crossword grids using constranit programming.

Related blog post: https://pedtsr.ca/2023/generating-crossword-grids-using-constraint-programming.html
"""


import sys

from ortools.sat.python import cp_model


def word_to_numbers(word):
    """Return the list of numbers associated with a word."""
    return list(' ABCDEFGHIJKLMNOPQRSTUVWXYZ'.find(letter)
                for letter in word.upper())


def load_words(path):
    """Load the words from a file and return a wordlist.

    The input file must have one word per line.

    Wordlist structure:

    {
      2: [[1, 25, 0], ..., [19, 7, 62]],
      5: [[16, 15, 7, 23, 18, 121], ..., [4, 8, 11, 10, 16, 1062]],
      ...
    }

    Note: The last item in each tuple is the word identifier.
    """
    with open(path, "r", encoding="utf-8") as f:
        words = list(word.strip() for word in f.readlines())

    wordlist = {}

    i = 0  # ID counter.

    for word in words:
        word_length = len(word)

        # We add the ID as an "extra" letter at the end of the word.
        if word_length in wordlist.keys():
            wordlist[word_length].append(word_to_numbers(word) + [i])

        else:
            wordlist[word_length] = [word_to_numbers(word) + [i]]

        i += 1  # Increment the ID for the next word.

    return wordlist


def table(model, variables, tuples, option):
    """Add an optional TABLE (AddAllowedAssignments()) constraint to a model.

    A TABLE constraint is added to `model`, where `variables` assume the values of one of `tuples`,
    but only when `option` is true.
    """
    # One boolean variable per tuple indicates if the values of that tuple are assigned to the
    # variables or not.
    b = [model.NewBoolVar(f"b[{i}]") for i in range(len(tuples))]

    # Set of values that can be assigned to the various variables.
    possible_values = {i: set() for i in range(len(variables))}
    max_value = max(max(t) for t in tuples)
    for t in tuples:
        for i, j in enumerate(t):
            possible_values[i].add(j)

    # is_assigned[i][j] indicates if variable `i` is assigned value `j`.
    is_assigned = [[model.NewBoolVar(f"is_assigned[{i}][{j}]")
                    for j in range(max_value+1)]
                   for i in range(len(variables))]

    # Some assignments are impossible since the value is found in no tuple.
    for i in range(len(variables)):
        for j in range(max_value+1):
            if j not in possible_values[i]:
                model.Add(is_assigned[i][j] == 0).OnlyEnforceIf(option)

    # One value must be assigned to each variable.
    for i in is_assigned:
        model.Add(cp_model.LinearExpr.Sum(i) == 1).OnlyEnforceIf(option)

    # Link `is_assigned` and `variables`.
    for i in range(len(variables)):
        for j in range(max_value+1):
            model.Add(variables[i] == j).OnlyEnforceIf(is_assigned[i][j])

    # TABLE constraint.
    for i, t in enumerate(tuples):
        model.AddBoolAnd([is_assigned[j][t[j]] for j in range(len(t))]).OnlyEnforceIf(b[i])

    # Only one tuple may be assigned to the variables.
    model.Add(cp_model.LinearExpr.Sum(b) == 1).OnlyEnforceIf(option)


wordlist = load_words("wordlist.txt")
rows = 8
columns = 8

model = cp_model.CpModel()

# L[r][c] denotes the letter at row `r` and column `c`.
# A = 1, B = 2, etc, and a black square is 0.
L = [[model.NewIntVar(0, 26, f"L[{r}][{c}]")
      for c in range(columns)]
     for r in range(rows)]

# B[r][c] == 1 if there is a black square at row `r` and column `c`, 0 otherwise.
B = [[model.NewBoolVar(f"B[{r}][{c}]")
      for c in range(columns)]
     for r in range(rows)]

# Link B and L.
for r in range(rows):
    for c in range(columns):
        model.Add(L[r][c] == 0).OnlyEnforceIf(B[r][c])
        model.Add(L[r][c] != 0).OnlyEnforceIf(B[r][c].Not())

# A[l][r][c] == 1 if an across word of length `l` starts at row `r` and column `c`, 0 otherwise.
A = [[[model.NewBoolVar(f"A[{l}][{r}][{c}]")
       for c in range(columns)]
      for r in range(rows)]
     for l in range(columns+1)]

# IA[r][c] denotes the identifier of the across word starting at row `r` and column `c`.
IA = [[model.NewIntVar(0, sum(len(wordlist[l]) for l in wordlist) + rows*columns*2, f"IA[{r}][{c}]")
       for c in range(columns)]
      for r in range(rows)]

# Across constraints. These also prevent any overlap between across words.
for l in range(columns+1):
    for r in range(rows):
        for c in range(columns):
            # A[l][r][c] == 0 if there are no words of length `l` in the word list.
            if l not in wordlist:
                model.Add(A[l][r][c] == 0)

            # A[l][r][c] == 0 if not enough squares remain on the right to fit the word.
            elif l > columns - c:
                model.Add(A[l][r][c] == 0)

            # The word fits in the row. If a word of length `l` starts at row `r` and column `c`, no
            # other word may start inside of it or adjacent to it. The word should also be bordered
            # on each side either by the grid edge or by a black square.
            else:
                # The word fills the whole row.
                if l == columns:
                    model.Add(cp_model.LinearExpr.Sum(list(A[i][r][c+j]
                                                           for i in range(columns+1)
                                                           for j in range(columns-c)))
                              == 1).OnlyEnforceIf(A[l][r][c])

                # The word starts on the left edge and does not fill the whole row.
                elif c == 0:
                    model.Add(cp_model.LinearExpr.Sum(list(A[i][r][c+j]
                                                           for i in range(columns+1)
                                                           for j in range(l+1)))
                              == 1).OnlyEnforceIf(A[l][r][c])

                # The word ends on the right edge and does not fill the whole row.
                elif c + l == columns:
                    model.Add(cp_model.LinearExpr.Sum(list(A[i][r][c+j]
                                                           for i in range(columns+1)
                                                           for j in range(columns-c)))
                              == 1).OnlyEnforceIf(A[l][r][c])

                # The word does not start or end on an edge.
                else:
                    model.Add(cp_model.LinearExpr.Sum(list(A[i][r][c-1+j]
                                                           for i in range(columns+1)
                                                           for j in range(l+2)))
                              == 1).OnlyEnforceIf(A[l][r][c])

                # Since `AddAllowedAssignments().OnlyEnforceIf()` is not (yet) supported, we have to
                # do it by hand.
                table(model, [L[r][c+i] for i in range(l)] + [IA[r][c]], wordlist[l], A[l][r][c])

# D[l][r][c] == 1 if a down word of length `l` starts at row `r` and column `c`, 0 otherwise.
D = [[[model.NewBoolVar(f"D[{l}][{r}][{c}]")
       for c in range(columns)]
      for r in range(rows)]
     for l in range(rows+1)]

# ID[r][c] denotes the identifier of the down word starting at row `r` and column `c`.
ID = [[model.NewIntVar(0, sum(len(wordlist[l]) for l in wordlist) + rows*columns*2, f"ID[{r}][{c}]")
       for c in range(columns)]
      for r in range(rows)]

# Down constraints. These also prevent any overlap between down words.
for l in range(rows+1):
    for r in range(rows):
        for c in range(columns):
            # D[l][r][c] == 0 if there are no words of length `l` in the word list.
            if l not in wordlist:
                model.Add(D[l][r][c] == 0)

            # D[l][r][c] == 0 if not enough squares remain on at the bottom to fit the word.
            elif l > rows - r:
                model.Add(D[l][r][c] == 0)

            # The word fits in the column. If a word of length `l` starts at row `r` and column `c`,
            # no other word may start inside of it or adjacent to it. The word should also be
            # bordered on each side either by the grid edge or by a black square.
            else:
                # The word fills the whole column.
                if l == rows:
                    model.Add(cp_model.LinearExpr.Sum(list(D[i][r+j][c]
                                                           for i in range(rows+1)
                                                           for j in range(rows-r)))
                              == 1).OnlyEnforceIf(D[l][r][c])

                # The word starts on the top edge and does not fill the whole column.
                elif r == 0:
                    model.Add(cp_model.LinearExpr.Sum(list(D[i][r+j][c]
                                                           for i in range(rows+1)
                                                           for j in range(l+1)))
                              == 1).OnlyEnforceIf(D[l][r][c])

                # The word ends on the bottom edge and does not fill the whole column.
                elif r + l == rows:
                    model.Add(cp_model.LinearExpr.Sum(list(D[i][r+j][c]
                                                           for i in range(rows+1)
                                                           for j in range(rows-r)))
                              == 1).OnlyEnforceIf(D[l][r][c])

                # The word does not start or end on an edge.
                else:
                    model.Add(cp_model.LinearExpr.Sum(list(D[i][r-1+j][c]
                                                           for i in range(rows+1)
                                                           for j in range(l+2)))
                              == 1).OnlyEnforceIf(D[l][r][c])

                # Since `AddAllowedAssignments().OnlyEnforceIf()` is not (yet) supported, we have to
                # do it by hand.
                table(model, [L[r+i][c] for i in range(l)] + [ID[r][c]], wordlist[l], D[l][r][c])

# Ensure that all words are different.
model.AddAllDifferent([IA[r][c]
                       for r in range(rows)
                       for c in range(columns)] +
                      [ID[r][c]
                       for r in range(rows)
                       for c in range(columns)])

# LLA[r][c] == 1 if the across letter at row `r` and column `c` is a lone letter, 0 otherwise.
LLA = [[model.NewBoolVar(f"LLA[{r}][{c}]")
        for c in range(columns)]
       for r in range(rows)]

# Constraints for LLA. A letter is a lone letter if it is bordered on each side with a grid edge or
# a black square.
for r in range(rows):
    # Edge-adjacent squares.
    model.Add(LLA[r][0] == B[r][1])
    model.Add(LLA[r][columns-1] == B[r][columns-2])

    # Other squares.
    for c in range(1, columns-1):
        model.Add(B[r][c-1] + B[r][c+1] == 2).OnlyEnforceIf(LLA[r][c])
        model.Add(B[r][c-1] + B[r][c+1] <= 1).OnlyEnforceIf(LLA[r][c].Not())

# LLAB[r][c] == 1 if the across letter at row `r` and column `c` is a lone letter or a black square,
# 0 otherwise.
LLAB = [[model.NewBoolVar(f"LLAB[{r}][{c}]")
         for c in range(columns)]
        for r in range(rows)]

# Constraints for LLAB.
for r in range(rows):
    for c in range(columns):
        model.Add(LLA[r][c] + B[r][c] >= 1).OnlyEnforceIf(LLAB[r][c])
        model.Add(LLA[r][c] + B[r][c] == 0).OnlyEnforceIf(LLAB[r][c].Not())

# If there is a letter in a square, and that this letter is not a lone letter, then an across word
# covering that letter must be active.
for r in range(rows):
    for c in range(columns):
        model.Add(cp_model.LinearExpr.Sum(list(A[i][r][c-j]
                                               for i in range(columns+1)
                                               for j in range(min(i, c+1))))
                  == 1).OnlyEnforceIf(LLAB[r][c].Not())

# LLD[r][c] == 1 if the down letter at row `r` and column `c` is a lone letter, 0 otherwise.
LLD = [[model.NewBoolVar(f"LLD[{r}][{c}]")
        for c in range(columns)]
       for r in range(rows)]

# Constraints for LLD. A letter is a lone letter if it is bordered on each side with a grid edge or
# a black square.
for c in range(columns):
    # Edge-adjacent squares.
    model.Add(LLD[0][c] == B[1][c])
    model.Add(LLD[rows-1][c] == B[rows-2][c])

    # Other squares.
    for r in range(1, rows-1):
        model.Add(B[r-1][c] + B[r+1][c] == 2).OnlyEnforceIf(LLD[r][c])
        model.Add(B[r-1][c] + B[r+1][c] <= 1).OnlyEnforceIf(LLD[r][c].Not())

# LLDB[r][c] == 1 if the down letter at row `r` and column `c` is a lone letter or a black square,
# 0 otherwise.
LLDB = [[model.NewBoolVar(f"LLDB[{r}][{c}]")
         for c in range(columns)]
        for r in range(rows)]

# Constraints for LLDB.
for r in range(rows):
    for c in range(columns):
        model.Add(LLD[r][c] + B[r][c] >= 1).OnlyEnforceIf(LLDB[r][c])
        model.Add(LLD[r][c] + B[r][c] == 0).OnlyEnforceIf(LLDB[r][c].Not())

# If there is a letter in a square, and that this letter is not a lone letter, then a down word
# covering that letter must be active.
for r in range(rows):
    for c in range(columns):
        model.Add(cp_model.LinearExpr.Sum(list(D[i][r-j][c]
                                               for i in range(rows+1)
                                               for j in range(min(i, r+1))))
                  == 1).OnlyEnforceIf(LLDB[r][c].Not())

# A letter shouldn't be a lone letter both across and down.
for r in range(rows):
    for c in range(columns):
        model.Add(LLA[r][c] + LLD[r][c] <= 1)

# Prevent 3x3 sub-grids from having too many black squares.
# This also prevents independent sub-grids.
if rows >= 3 and columns >= 3:
    for r in range(0, rows-2):
        for c in range(0, columns-2):
            model.Add(cp_model.LinearExpr.Sum(list(B[i][j]
                                                   for i in range(r, r+3)
                                                   for j in range(c, c+3)))
                      <= 2)

# Limit the number of black squares.
model.Add(cp_model.LinearExpr.Sum(
    list(cp_model.LinearExpr.Sum(
        list(c for c in row)) for row in B))
          <= int(rows*columns/5))  # int(x) is the same as floor(x)

solver = cp_model.CpSolver()
status = solver.Solve(model)
status_name = solver.StatusName(status)
print(f"Status: {status_name}")
print()

if status_name != "OPTIMAL":
    print("We have a problem :(")
    sys.exit()

for r in range(rows):
    for c in range(columns):
        print('.ABCDEFGHIJKLMNOPQRSTUVWXYZ'[solver.Value(L[r][c])], end="")
    print()
