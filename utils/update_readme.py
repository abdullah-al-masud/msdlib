"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

from xml.dom import minidom
import subprocess
import os


readme_paths = ['README.md', 'docs/README.rst']


def generate_coverage_xml():
    if '.coverage' in os.listdir('.'):
        out = subprocess.run('coverage xml', stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT).returncode
        print('return code for "coverage xml"', out)
    else:
        out = -1
    return out


def get_coverage():
    dom = minidom.parse('coverage.xml')
    elements = dom.getElementsByTagName('coverage')

    if len(elements) > 0:
        for element in elements:
            percentage = float(element.attributes['line-rate'].value) * 100
            break
    else:
        raise ValueError('coverage.xml file doesnt contain proper coverage report')

    return percentage


def categorize_coverage_percentage(perc):
    if perc >= 90:
        color = 'lightgreen'
    elif perc >= 70 and perc < 90:
        color = 'green'
    elif perc >= 50 and perc < 70:
        color = 'yellow'
    elif perc >= 0 and perc < 50:
        color = 'red'
    else:
        raise ValueError('coverage percentage %f is not a valid percentage'%(perc))
    
    return color


def edit_readme_with_coverage(readme_paths, perc, color):
    for path in readme_paths:
        with open(path, 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if 'https://img.shields.io/badge/coverage' in line:
                if path.endswith('.md'):
                    line = '![](https://img.shields.io/badge/coverage-%d%%25-%s)\n'%(perc, color)
                    lines[i] = line
                elif path.endswith('.rst'):
                    line = '.. image:: https://img.shields.io/badge/coverage-%d%%25-%s\n'%(perc, color)
                    lines[i] = line
                break

        with open(path, 'w') as file:
            file.writelines(lines)
    
    os.remove('coverage.xml')


def update_code_coverage(readme_paths):
    code = generate_coverage_xml()
    if code == 0:
        perc = get_coverage()
        color = categorize_coverage_percentage(perc)
        print('code coverage percentage:', round(perc), ',  badge color-', color)
        edit_readme_with_coverage(readme_paths, round(perc), color)
    elif code == -1:
        print('.coverage file is not available, skipping code coverage update')
    else:
        print('couldnt generate coverage xml file, skipping code coverage update')


if __name__ == '__main__':
    update_code_coverage(readme_paths)
