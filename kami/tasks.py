from invoke import task

@task
def hello(c):
    """hello world."""
    print("Hello, world!")
