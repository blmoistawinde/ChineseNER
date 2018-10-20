import api
def main():
    # 从文件(UTF-8编码)中按行进行NER
    print(api.evaluate_lines_file("lines.txt"))

    with open("lines.txt", "r", encoding="utf-8") as f:
        lines = f.read().split()
    lines = ["曹操出生于一个显赫的官宦家庭。",
            "曹操的祖父曹腾，是东汉末年宦官集团中的一员，汉相国曹参的后人。",
            "父亲曹嵩，是曹腾的养子。"]
    # 从列表中按行进行NER
    for result in api.evaluate_lines(lines):
        print(result)
if __name__ == "__main__":
    main()
