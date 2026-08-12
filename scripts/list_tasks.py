"""List scaffolded task IDs and viewer scripts for this project."""

TASKS = ["HumanoidSoccer-Direct-v0"]
VIEWERS = ["scripts/view_nao.py"]


def main() -> None:
    for task in TASKS:
        print(task)
    for viewer in VIEWERS:
        print(viewer)


if __name__ == "__main__":
    main()
