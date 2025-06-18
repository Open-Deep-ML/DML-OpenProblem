In this problem, you need to implement a function that calculates the [Optimal String Alignment (OSA)](https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance) distance between two given strings. The OSA distance represents the minimum number of edits required to transform one string into another. The allowed edit operations are:

- Insert a character
- Delete a character
- Substitute a character
- Transpose two adjacent characters

Each of these operations costs 1 unit.

Your task is to find the minimum number of edits needed to convert the first string (s1) into the second string (s2).

For example, the OSA distance between the strings `caper` and `acer` is 2: one deletion (removing "p") and one transposition (swapping "a" and "c").
