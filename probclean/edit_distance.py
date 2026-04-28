def levenshtein(s1: str, s2: str) -> int:
  if s1 == s2:
    return 0
  if not s1:
    return len(s2)
  if not s2:
    return len(s1)
  #Ensure s2 is the shorter string for space optimization
  if len(s1) < len(s2):
        s1, s2 = s2, s1

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insert_cost = curr[j] + 1
            delete_cost = prev[j + 1] + 1
            replace_cost = prev[j] + (0 if c1 == c2 else 1)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev = curr

    return prev[-1]

def normalized_smiliarity(s1: str, s2: str) -> float:
  if s1 == s2:
    return 1.0
  max_len = max(len(s1), len(s2))
  if max_len == 0:
    return 1.0
  dist = levenshtein(s1,s2)
  return 1.0 - dist/ max_len
