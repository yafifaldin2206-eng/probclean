def levenshtein(s1: str, s2: str) -> int:
  if s1 == s2:
    return 0
  if not s1:
    return len(s2)
  if not s2:
    return len(s1)
