
## Optimal String Alignment Distance

Given two strings \( s_1 \) and \( s_2 \), find the Optimal String Alignment (OSA) distance between them.

The OSA distance gives the minimum number of edits needed to transform string \( s_1 \) into \( s_2 \). Here are the allowed edit operations:

1. **Insert a character**
2. **Delete a character**
3. **Substitute a character**
4. **Transpose two adjacent characters**

Each operation has a cost of 1 unit.

For example, the OSA distance between the strings "caper" and "acer" is 2:
- One deletion (removing the letter 'p')
- One transposition (swapping the letters 'a' and 'c')
