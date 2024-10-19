# استيراد المكتبات اللازمة
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

# استيراد البيانات (تأكد من أنك تقوم بتحميل البيانات هنا)
# final_data = pd.read_csv('path_to_your_data.csv')  # قم بإلغاء تعليق هذه السطر و تعديل المسار
# استخدم البيانات الخاصة بك

# إعداد الواجهة
st.title("تقييم نموذج Random Forest")
st.write("هذا التطبيق يستخدم لتقييم نموذج Random Forest على مجموعة بيانات.")

# قم بتحميل البيانات
uploaded_file = st.file_uploader("تحميل ملف CSV", type='csv')

if uploaded_file is not None:
    # قراءة البيانات
    final_data = pd.read_csv(uploaded_file)

    # إسقاط الأعمدة غير الضرورية
    features = final_data.drop(columns=['satisfaction'])
    target = final_data['satisfaction']

    # تقسيم البيانات إلى تدريب واختبار
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)

    # دالة لتقييم النموذج
    def evaluate_model(model, x_train, x_test, y_train, y_test):
        model.fit(x_train, y_train)
        train_acc = round(model.score(x_train, y_train) * 100, 2)
        test_acc = round(model.score(x_test, y_test) * 100, 2)
        test_pred = model.predict(x_test)
        test_con_mat = confusion_matrix(y_test, test_pred)

        return train_acc, test_acc, test_pred, test_con_mat

    # استخدام نموذج RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=50, random_state=0)

    # تقييم النموذج
    train_acc, test_acc, test_pred, test_con_mat = evaluate_model(rf, x_train, x_test, y_train, y_test)

    # عرض النتائج
    st.subheader("نتائج النموذج")
    st.write(f"نسبة دقة التدريب: {train_acc}%")
    st.write(f"نسبة دقة الاختبار: {test_acc}%")
    
    st.subheader("مصفوفة الارتباك")
    st.write(test_con_mat)

    # طباعة تقرير التصنيف
    st.subheader("تقرير التصنيف")
    report = classification_report(y_test, test_pred, output_dict=True)
    st.dataframe(report)

# تنفيذ التطبيق
if __name__ == '__main__':
    st.write("قم بتحميل البيانات أعلاه لبدء تقييم النموذج.")
