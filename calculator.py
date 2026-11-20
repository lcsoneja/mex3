import tkinter as tk
from tkinter import messagebox


class Calculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculator")
        self.root.geometry("300x400")
        self.root.resizable(False, False)

        self.expression = ""
        self.input_text = tk.StringVar()

        # Display
        display_frame = tk.Frame(self.root, width=300, height=50, bd=0)
        display_frame.pack(side=tk.TOP)

        display = tk.Entry(
            display_frame,
            font=("arial", 24, "bold"),
            textvariable=self.input_text,
            width=50,
            bg="#eee",
            bd=0,
            justify=tk.RIGHT,
        )
        display.grid(row=0, column=0)
        display.pack(ipady=10)

        # Buttons
        btns_frame = tk.Frame(self.root, width=300, height=350, bg="grey")
        btns_frame.pack()

        buttons = [
            ("C", 0, 0, self.clear, "#ff6666"),
            ("/", 0, 1, lambda: self.click("/"), "#cccccc"),
            ("*", 0, 2, lambda: self.click("*"), "#cccccc"),
            ("7", 1, 0, lambda: self.click("7"), "#ffffff"),
            ("8", 1, 1, lambda: self.click("8"), "#ffffff"),
            ("9", 1, 2, lambda: self.click("9"), "#ffffff"),
            ("-", 1, 3, lambda: self.click("-"), "#cccccc"),
            ("4", 2, 0, lambda: self.click("4"), "#ffffff"),
            ("5", 2, 1, lambda: self.click("5"), "#ffffff"),
            ("6", 2, 2, lambda: self.click("6"), "#ffffff"),
            ("+", 2, 3, lambda: self.click("+"), "#cccccc"),
            ("1", 3, 0, lambda: self.click("1"), "#ffffff"),
            ("2", 3, 1, lambda: self.click("2"), "#ffffff"),
            ("3", 3, 2, lambda: self.click("3"), "#ffffff"),
            ("=", 3, 3, self.equal, "#66b3ff"),
            ("0", 4, 0, lambda: self.click("0"), "#ffffff"),
            (".", 4, 1, lambda: self.click("."), "#ffffff"),
        ]

        for text, row, col, command, color in buttons:
            btn = tk.Button(
                btns_frame,
                text=text,
                width=10,
                height=3,
                font=("arial", 14),
                fg="black",
                bg=color,
                cursor="hand2",
                bd=0,
                command=command,
            )
            btn.grid(row=row, column=col, padx=1, pady=1)

    def click(self, item):
        self.expression = self.expression + str(item)
        self.input_text.set(self.expression)

    def clear(self):
        self.expression = ""
        self.input_text.set("")

    def equal(self):
        try:
            result = str(eval(self.expression))
            self.input_text.set(result)
            self.expression = result
        except Exception:
            self.input_text.set("error")
            self.expression = ""


if __name__ == "__main__":
    root = tk.Tk()
    Calculator(root)
    root.mainloop()
