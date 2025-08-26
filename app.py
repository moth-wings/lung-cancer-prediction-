from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import joblib
import numpy as np
import pandas as pd
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import cv2
import os

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/images'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load mô hình
model = joblib.load('lung_cancer_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/process', methods=['POST'])
def process():
    text = request.form.get('text', '').lower()
    
    if "hiện biểu mẫu" in text:
        return jsonify({'redirect': url_for('form')})
    
    responses = {
        "hello": "Xin chào bạn, tôi có thể giúp gì được cho bạn?",
        "có thể giúp gì cho tôi" : "Tôi có thể tư vấn về các bệnh liên quan đến phổi ",
        "chính xác không" : "Uy tín 100% ",
        "triệu chứng":"Bạn có thể mô tả chi tiết các triệu chứng mà bạn gặp phải hay không !",
        "chào": "Xin chào bạn, tôi có thể giúp gì được cho bạn?",
        "hi": "Xin chào bạn, tôi có thể giúp gì được cho bạn?",
        "bye": "Tạm biệt bạn, cảm ơn bạn đã sử dụng dịch vụ của chúng tôi!",
        "tạm biệt": "Tạm biệt bạn, cảm ơn bạn đã sử dụng dịch vụ của chúng tôi!",
        "công việc của bạn": "Công việc của tôi là tư vấn về tình hình bệnh của bạn và giúp bạn phòng chống căn bệnh ung thư phổi.",
        "bạn có thể giúp gì được cho tôi": "Công việc của tôi là tư vấn về tình hình bệnh của bạn và giúp bạn phòng chống căn bệnh ung thư phổi.",
        "tư vấn": "Bạn có thể nêu rõ tình hình bệnh lý của bạn được không?",
        "tình hình cơ bản": "Bạn có thể nêu rõ tình hình bệnh lý của bạn được không?",
        "bạn tên gì": "Tôi là Robot AI được phát triển và vận hành bởi nhóm học Trí tuệ nhân tạo D18CN01.",
        "đau ngực": "Bạn có từng đi khám về vấn đề này chưa?",
        "khó thở": "Bạn có triệu chứng này khi vận động nhẹ không? Bạn có tiền sử hút thuốc không?",
        "ho khan": "Ho của bạn kéo dài bao lâu? Có đờm hoặc máu không?",
        "sụt cân": "Bạn có sụt cân không rõ lý do không? Đây có thể là dấu hiệu liên quan đến phổi.",
        "mệt mỏi": "Cảm giác mệt mỏi của bạn có kéo dài không? Bạn có chóng mặt hay mất tập trung không?",
        "sốt nhẹ": "Bạn có sốt về chiều hoặc đêm không?",
        "đau vai": "Bạn có thấy đau tăng khi hít thở sâu không?",
        "đờm có máu": "Bạn nên đến cơ sở y tế ngay để kiểm tra.",
        "hút thuốc": "Bạn hút bao lâu và bao nhiêu điếu mỗi ngày? Đã bỏ thuốc chưa?",
        "tiền sử gia đình": "Gia đình bạn có ai mắc ung thư phổi không?",
        "rồi": "Kết quả bệnh án của bạn thế nào?",
        "viêm phổi": "Kết quả này đã lâu chưa? Bạn có làm xét nghiệm nào không?",
        "tôi nên làm gì": "Bạn nên đi bệnh viện để làm xét nghiệm hình ảnh và máu.",
        "khám ở đâu": "Bạn có thể đến bệnh viện hô hấp hoặc ung bướu gần nhất.",
        "hà nội": "1. Bệnh viện Quân đội 103\n2. Bệnh viện Phổi Hà Nội\n3. Khoa hô hấp Bệnh viện E\n4. Trung tâm Nội Hô hấp - Bệnh viện Quân Y 103",
        "thành phố hồ chí minh": "Bệnh viện Chợ Rẫy",
        "chắc chắn hơn": "Bạn có thể điền Form bằng cách nhấn vào nút [Điền biểu mẫu] để hiểu rõ hơn về các vấn đề của bạn",
        'không chắc chắn': 'Bạn có thể điền Form bằng cách nhấn vào nút [Điền biểu mẫu] để hiểu rõ hơn về các vấn đề của bạn',
        "form": "Bạn có thể điền Form bằng cách nhấn vào nút [Điền biểu mẫu] để hiểu rõ hơn về các vấn đề của bạn",
        "biểu mẫu": "Bạn có thể điền Form bằng cách nhấn vào nút [Điền biểu mẫu] để hiểu rõ hơn về các vấn đề của bạn",
        "4 vấn": "Bạn có thể nêu rõ tình hình bệnh lý của bạn được không?",
        'có': 'Mức độ nhiễm bệnh về phổi của bạn rất cao, bạn có thể đi khám ở các bệnh viện uy tín về phổi ở Hà Nội hoặc Thành Phố Hồ Chí Minh, hãy cho tôi biết bạn muốn đến đâu trong hai tỉnh trên',
        "triệu chứng": "Bạn có thể mô tả chi tiết các triệu chứng của bạn không?",
        "điều trị": "Bạn đã từng điều trị bệnh này chưa? Nếu có, bạn đã điều trị như thế nào?",
        "thuốc": "Bạn có đang sử dụng loại thuốc nào không? Nếu có, vui lòng cho biết tên thuốc.",
        "dị ứng": "Bạn có bị dị ứng với bất kỳ loại thuốc hoặc chất nào không?",
        "tiền sử bệnh": "Bạn có tiền sử bệnh lý nào khác không?",
        "bệnh viện": "Bạn có thể đến bệnh viện gần nhất để kiểm tra và điều trị.",
        "bác sĩ": "Bạn nên gặp bác sĩ chuyên khoa để được tư vấn và điều trị.",
        "xét nghiệm": "Bạn đã làm xét nghiệm nào chưa? Nếu có, vui lòng cho biết kết quả.",
        "chụp X-quang": "Bạn đã chụp X-quang chưa? Nếu có, vui lòng tải lên hình ảnh để tôi có thể phân tích.",
        "MRI": "Bạn đã làm MRI chưa? Nếu có, vui lòng tải lên hình ảnh để tôi có thể phân tích.",
        "CT scan": "Bạn đã làm CT scan chưa? Nếu có, vui lòng tải lên hình ảnh để tôi có thể phân tích.",
        "khám sức khỏe": "Bạn nên đi khám sức khỏe định kỳ để phát hiện sớm các vấn đề về sức khỏe.",
        "tư vấn trực tuyến": "Bạn có thể tư vấn trực tuyến với bác sĩ chuyên khoa để được hỗ trợ.",
        "hỗ trợ": "Tôi có thể hỗ trợ bạn về các vấn đề liên quan đến sức khỏe. Vui lòng cho biết chi tiết.",
        "liên hệ": "Bạn có thể liên hệ với chúng tôi qua số điện thoại hoặc email để được hỗ trợ.",
        "địa chỉ": "Vui lòng cung cấp địa chỉ của bạn để chúng tôi có thể hỗ trợ tốt hơn.",
        "tình trạng": "Vui lòng mô tả tình trạng hiện tại của bạn để tôi có thể tư vấn chính xác hơn.",
        "triệu chứng khác": "Bạn có triệu chứng nào khác không? Vui lòng mô tả chi tiết.",
        "bệnh lý": "Bạn có tiền sử bệnh lý nào khác không? Vui lòng cho biết chi tiết.",
        "chế độ ăn uống": "Chế độ ăn uống của bạn như thế nào? Bạn có ăn uống đầy đủ và cân đối không?",
        "lối sống": "Lối sống của bạn như thế nào? Bạn có thường xuyên tập thể dục không?",
        "nghỉ ngơi": "Bạn có nghỉ ngơi đầy đủ không? Giấc ngủ của bạn có đủ và chất lượng không?",
        "căng thẳng": "Bạn có bị căng thẳng không? Nếu có, bạn đã làm gì để giảm căng thẳng?",
        "hút thuốc lá": "Bạn có hút thuốc lá không? Nếu có, bạn hút bao nhiêu điếu mỗi ngày?",
        "uống rượu": "Bạn có uống rượu không? Nếu có, bạn uống bao nhiêu ly mỗi ngày?",
        "Tôi cần làm gì":"Bạn cần theo dõi các triệu chứng và tham khảo ý kiến bác sĩ để được tư vấn và điều trị kịp thời.",
        "ung thư phổi": "Ung thư phổi là một loại ung thư bắt đầu trong phổi, thường do hút thuốc hoặc tiếp xúc với các chất độc hại.",
        "triệu chứng": "Các triệu chứng bao gồm ho kéo dài, khó thở, đau ngực, sụt cân không rõ nguyên nhân, và ho ra máu.",
        "phòng ngừa": "Bạn có thể phòng ngừa bằng cách không hút thuốc, tránh tiếp xúc với khói thuốc, ăn uống lành mạnh và tập thể dục thường xuyên.",
       
        }

    response = next((responses[key] for key in responses if key in text),
                    "Có vẻ như bệnh của bạn không nghiêm trọng hoặc không liên quan đến ung thư phổi. Để chắc chắn, bạn nên đi khám.")
    
    return jsonify({'response': response})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': "Không có tệp nào được tải lên."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': "Vui lòng chọn một tệp để tải lên."}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
  
    if not os.path.exists(filepath):
        return jsonify({'error': "Tệp không tồn tại."}), 400
    
    img = cv2.imread(filepath, 1)
    if img is None:
        return jsonify({'error': "Không thể đọc tệp ảnh."}), 400
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    dark_pixels = np.sum(gray == 0)
    bright_pixels = np.sum(gray == 255)
    

    ratio = bright_pixels / dark_pixels if dark_pixels != 0 else 'Không xác định'
    
    if isinstance(ratio, str):
        result = "Nguy cơ mắc ung thư phổi rất cao. Hãy tham khảo ý kiến bác sĩ chuyên khoa ngay lập tức để thực hiện kiểm tra chi tiết hơn."
    elif ratio > 0.26:
        result = "Nguy cơ mắc ung thư phổi rất cao. Bạn nên đi khám bác sĩ sớm để có thể phát hiện và điều trị kịp thời."
    elif 0.15 < ratio < 0.2:
        result = "Có khả năng cao có bệnh về phổi. Hãy theo dõi các triệu chứng như ho kéo dài, khó thở và tham khảo ý kiến bác sĩ nếu cần."
    else:
        result = "Không có khả năng mắc bệnh. Tuy nhiên, hãy duy trì lối sống lành mạnh và kiểm tra sức khỏe định kỳ để đảm bảo sức khỏe tốt nhất."
    
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc ra hai vùng lớn nhất (thường là hai lá phổi)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius * 1.2)  # Tăng kích thước hình tròn để khoanh vùng phổi rõ hơn
        cv2.circle(img, center, radius, (0, 0, 255), 2)
    
    # Lưu ảnh đã xử lý
    processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file.filename)
    cv2.imwrite(processed_filepath, img)
    
    return jsonify({
        'result': result,
        'image_url': url_for('uploaded_file', filename='processed_' + file.filename)
    })

@app.route('/static/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array([
            data['age'], data['smoking'], data['chronic_cough'], 
            data['breath_difficulty'], data['chest_pain'], data['family_history']
        ]).reshape(1, -1)
        
        prediction = model.predict(features)[0]
        
        if prediction == 1:
            result = "Nguy cơ mắc ung thư phổi cao (Nguy kịch)"
        elif prediction == 2:
            result = "Nguy cơ mắc ung thư phổi trung bình"
        elif prediction == 3:
            result = "Nguy cơ mắc ung thư phổi thấp"
        else:
            result = "Không có nguy cơ mắc ung thư phổi"
        
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)