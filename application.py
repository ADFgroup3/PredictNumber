from flask import Flask,render_template,request
import PIL.Image
import PIL.ImageTk
#機械学習で使うモジュール
import sklearn.datasets
import sklearn.svm
import numpy

app = Flask(__name__)

#画像ファイルを数値リストに変換する
def imageToData(filename):
    #画像を8x8のグレースケールに変換
    grayImage = PIL.Image.open(filename).convert("L")
    grayImage = grayImage.resize((8,8),PIL.Image.ANTIALIAS)
    #数値リストに変換する
    numImage = numpy.asarray(grayImage, dtype = float)
    numImage = numpy.floor(16 -16 * (numImage / 256))
    numImage = numImage.flatten()
    return numImage

#数字を予測する
def predictDigits(data):
    #学習用データを読み込む
    digits = sklearn.datasets.load_digits()
    #機械学習をする
    clf = sklearn.svm.SVC(gamma=0.001)
    clf.fit(digits.data, digits.target)
    #予測結果を表示する
    n = clf.predict([data])
    return n


@app.route("/", methods=["GET", "POST"])
def main_page():
    if request.method == 'GET':
        text = "ここに結果が出力されます"
        return render_template("page.html",text=text)

    elif request.method == 'POST':
        image = request.files['image']
        data = imageToData(image)
        text = predictDigits(data)
        return render_template("page.html",text=text)


## 実行
if __name__ == "__main__":
    app.run(debug=True)