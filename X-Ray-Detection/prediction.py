from flask import Flask, render_template, request, redirect, url_for, flash

from db_handling import insert_data, select_data, delete_data, update_password
from pneumonia_prediction import predict_probability

app = Flask(__name__)

app.secret_key = 'flask'

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/admin', methods = ['GET', 'POST'])
def admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            return redirect(url_for('admin_panel'))
        else:
            flash('Invalid username or password')
            return render_template('admin.html')
    else:
        return render_template('admin.html')


@app.route('/admin_panel')
def admin_panel():
    connection, cur, c = select_data()
    datas = []
    for i in c:
        datas.append(i)
    return render_template('admin_panel.html', datas=datas)


@app.route('/signup', methods = ['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['newpass']
        res = insert_data(username, password)
        print(res)
        if res is False:
            err = "The password you entered already exists Try with different Password"
            flash(err)
            return redirect(url_for('signup'))
        
        err = "New User Sign Up Successful"
        flash(err)
        return redirect(url_for('login'))
        
    return render_template('signup.html')


@app.route('/login', methods = ['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['pass']
        i, j, curr = select_data(username, password)
        count = 0
        for k in curr:
            count+=1
        if count > 0:
            return redirect(url_for('pneumonia_detection'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    else:
        return render_template('login.html')


@app.route('/sample_files', methods=['POST', 'GET'])
def sample_files():
    if request.method == 'POST':
        image_name = request.form['image']
        image_file = image_name
        image_name = 'static/'+image_name 
        pred_name = predict_probability(image_name)
        
        print(image_name, image_file)
        return render_template('predict.html', prediction=pred_name, image_file=image_file)

    return render_template('sample_files.html')


@app.route('/pneumonia_detection')
def pneumonia_detection():
    return render_template('index.html')


@app.route('/delete_data', methods=['GET', 'POST'])
def delete_records():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        delete_data(username, password)
        redirect(url_for('admin_panel'))
    redirect(url_for('admin_panel'))
    # return render_template('admin_panel.html')


@app.route('/forgot', methods = ['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['newpass']
        update_password(username, password)
        redirect(url_for('login'))
    return render_template('forgot.html')



@app.route('/pneumonia_detection/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    image_name = '/images/img.png'
    image_file = f'./static{image_name}'
    img_file.save(image_file)
    pred_name = predict_probability(image_file)
    return render_template('predict.html', prediction=pred_name, image_file=image_name)

if __name__ == '__main__':
    # app.run(debug=True)
    
    app.run(debug=True, host='0.0.0.0')
