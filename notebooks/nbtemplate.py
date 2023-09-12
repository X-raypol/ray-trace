'''This module collects utilities for the RedSpx notebooks.'''
import sys
import subprocess
import re
import logging

from IPython.display import display, Markdown

__all__ = ['display_header']

class DisableLogger():
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, a, b, c):
       logging.disable(logging.NOTSET)


reexp = re.compile(r"(?P<version>[\d.dev]+[\d]+)[+]?(g(?P<gittag>\w+))?[.]?(d(?P<dirtydate>[\d]+))?")


def parse_git_scm_version(version):
    ver = reexp.match(version)
    v = ver.group('version')
    if not ver.group('gittag') is None:
        v += ' (commit hash: ' + ver.group('gittag') + ')'
    if not ver.group('dirtydate') is None:
        v += ' (Date of dirty version: ' + ver.group('dirtydate') + ')'
    return v


def get_marxs_status():
    try:
        import marxs.version as mvers
    except ImportError:
        return 'MARXS cannot be imported. No version information is available.'
    return 'MARXS ray-trace code version ' + parse_git_scm_version(mvers.version)


def get_nb_status(filename):
    try:
        gitlog = subprocess.check_output(['git',  'log', '-1', '--use-mailmap',
                                          '--format=medium',  '--', filename])
        gitlog = gitlog.decode(sys.stdout.encoding)
    except subprocess.CalledProcessError:
        return '''git is not installed or notebook was run outside of git version control.
        No versioning information can be displayed.'''

    if len(gitlog) == 0:
        return '''file: {} not found in repository (path missing or new file not yet committed?).
        No versioning information can be displayed.'''.format(filename)
    else:
        gitlog = gitlog.split('\n')
        out = '''Last revision in version control:

- {1}
- {0}
- {2}
'''.format(gitlog[0], gitlog[1], gitlog[2])
        modifiedfiles = subprocess.check_output(['git', 'ls-files', '-m'])
        modifiedfiles = modifiedfiles.decode(sys.stdout.encoding)
        if filename in modifiedfiles:
            out = out + '''
**The version shown here is modified compared to the last committed revision.**

            '''
    return out


def revision_status(filename, status=None):
    if status is None:
        statusstring = ''
    else:
        statusstring = ': *{}*'.format(status)
    out='### Revision status{}\n'.format(statusstring)

    out = out + '''{nbstatus}

This document is git version controlled. The repository is available at https://github.com/X-raypol/ray-trace.
See git commit log for full revision history.

Code was last run with:

- {marxs}
'''.format(nbstatus=get_nb_status(filename),
           marxs=get_marxs_status())
    return Markdown(out)


def display_header(filename, status=None):
    display(revision_status(filename, status=status))
