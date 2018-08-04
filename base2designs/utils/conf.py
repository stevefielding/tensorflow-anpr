# import the necessary packages
import json
import re

class Conf:
  def __init__(self, confPath):
    # load and store the configuration and update the object's dictionary
    stripped = re.sub(r'(.*?)#.*', r'\1', open(confPath).read())
    conf = json.loads(stripped)
    self.__dict__.update(conf)

  def __getitem__(self, k):
    # return the value associated with the supplied key
    val = self.__dict__.get(k, None)
    if val == None:
      print("[ERROR] Attempting to get undefined config val conf[\"{}\"]".format(k))
    return self.__dict__.get(k, None)
