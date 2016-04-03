import subprocess


def create_html(file):
    command = ["jupyter", "nbconvert", file]
    subprocess.call(command)

def run_notebook(notebook_file,outfile):
    command = ["runipy", notebook_file, outfile]
    subprocess.call(command)

if __name__ == "__main__":
    notebook_file = "AdaBoost_Hackathon-test.ipynb"
    outfile = "result_notebook.ipynb"
    run_notebook(notebook_file,outfile)
    create_html(outfile)