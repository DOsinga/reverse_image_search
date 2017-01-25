import time
import fcntl, termios, struct
from collections import OrderedDict
import math

def numstr(number):
  exp = min(int(math.log10(number) / 3), 4)
  number = ('%g' % (number / (1e3 ** exp)))[:4].strip('.')
  return number + ' KMGT'[exp].strip()

class Status(OrderedDict):
  """Simple object to report the status of a command line tool in the form of counters."""

  def __init__(self, qps_field=None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._colwidths = {}
    self._count = 0
    self._first_ts = time.time()
    self._qps_field = qps_field

  def count(self, tag, increment=1):
    self[tag] = self.get(tag, 0) + increment
    return self

  def report(self):
    self._count += 1
    formatted = [(k, numstr(v)) for k, v in self.items()]
    self._colwidths = {k: max(self._colwidths.get(k, 0), len(v)) for k, v in formatted}
    repr = ['%s=%s%s' % (k, v, ' ' * (self._colwidths[k] - len(v))) for k, v, in formatted]
    if self._qps_field and self._qps_field in self:
      repr.insert(0, '%2.2f/s' % (self[self._qps_field] / (time.time() - self._first_ts)))
    line = ' '.join(['◐◓◑◒'[self._count % 4]] + repr)
    try:
      _, width = struct.unpack('hh', fcntl.ioctl(1, termios.TIOCGWINSZ, '1234'))
      if len(line) > width:
        line = line[:width - 4] + '...'
    except OSError:
      pass
    print('\r' + line, end='')
    return self