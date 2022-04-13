class Test:

  var_a = "var_a"

  def __init__(self) -> None:
    """
    """
    self.abc = 10

  def do(self, a: int, b: str="hello") -> bool:
    """_summary_

    Args:
      a (int): _description_
      b (str, optional): _description_. Defaults to "hello".

    Returns:
      bool: _description_
    """
    print(self.abc)
    return 10
