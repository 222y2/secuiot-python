import os
from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
from supabase import create_client, Client
from pydantic import BaseModel
import pickle
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# .env 파일 로드
# load_dotenv()

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://secuiot.vercel.app"],  # React 앱의 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase 클라이언트 초기화
url: str = os.getenv("https://pabwlxbmxwkjxfepnjax.supabase.co")
key: str = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBhYndseGJteHdranhmZXBuamF4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyNjkyMDUyOSwiZXhwIjoyMDQyNDk2NTI5fQ.HXJvSvsDFAPN1fKv4UbtesUtw7ruOaQSVfkd6ytJRZw")
supabase: Client = create_client(url, key)

# 현재 파일의 디렉토리 경로
current_dir = os.path.dirname(__file__)

# model.pkl 파일의 상대경로를 설정
model_path = os.path.join(current_dir, 'api', 'model.pkl')

# 모델 로드
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
except:
    # pickle로 로드가 안 되면 joblib으로 시도
    model = joblib.load(model_path)

print(f"모델 타입 : {type(model)}")
print(f"모델 : {model}")


# 레이블 인코더 초기화 및 훈련
device_encoder = LabelEncoder()
carrier_encoder = LabelEncoder()

# Supabase에서 기존 데이터를 가져와 LabelEncoder를 훈련시킵니다.
def init_label_encoders():
    try:
        devices_response = supabase.table("devices").select("device_name, carrier").execute()
        devices_data = devices_response.data

        device_names = [device['device_name'] for device in devices_data]
        carriers = [device['carrier'] for device in devices_data]

        # 고유한 값들로 LabelEncoder 훈련
        device_encoder.fit(list(set(device_names)))
        carrier_encoder.fit(list(set(carriers)))

        print("LabelEncoders initialized successfully")
    except Exception as e:
        print(f"Error initializing LabelEncoders: {str(e)}")

# 애플리케이션 시작 시 LabelEncoder 초기화
init_label_encoders()

# 라우터 생성
router = APIRouter()

class DeviceCreate(BaseModel):
    device_name: str
    carrier: str


# 전역 변수로 risk_levels 정의
risk_levels = {
    "level_very_safe": {
        "name": "VERY SAFE",
        "min_score": 0,
        "max_score": 5,
        "color": "#0084ff",
        "description": "매우 안전합니다."
    },
    "level_safe": {
        "name": "SAFE",
        "min_score": 6,
        "max_score": 10,
        "color": "#05ff00",
        "description": "안전합니다."
    },
    "level_caution": {
        "name": "CAUTION",
        "min_score": 11,
        "max_score": 15,
        "color": "#fff400",
        "description": "높은 수준의 취약점, CVE 기록, 악성 활동 신고가 있으며, 적극적인 조치가 필요한 상태입니다. 추가 검사와 보안 시스템 강화 등의 조치가 필요합니다. 네트워크 보안 시스템 점검을 추천드립니다."
    },
    "level_dangerous": {
        "name": "DANGEROUS",
        "min_score": 16,
        "max_score": 20,
        "color": "#ff5b00",
        "description": "위험이 인식되었습니다. 추가 검사와 보안 시스템 강화 등의 조치가 필요합니다. 네트워크 보안 시스템 점검을 추천드립니다."
    },
    "level_very_dangerous": {
        "name": "VERY DANGEROUS",
        "min_score": 21,
        "max_score": float('inf'),
        "color": "#ff0000",
        "description": "매우 위험합니다. 추가 검사와 보안 시스템 강화 등의 조치가 필요합니다. "
    }
}

@router.post("/devices")
async def create_device(device: DeviceCreate, request: Request):  # Request 파라미터 추가
    try:
        # 클라이언트 IP 주소 가져오기
        client_ip = request.client.host

        # Supabase에 기기명과 통신사 저장
        response = supabase.table("devices").insert({
            "device_name": device.device_name,
            "carrier": device.carrier
        }).execute()

        # 입력 데이터 전처리
        try:
            if device.device_name and device.carrier:
                # 새로운 레이블이 들어왔을 때 LabelEncoder 업데이트
                if device.device_name not in device_encoder.classes_:
                    device_encoder.classes_ = np.append(device_encoder.classes_, device.device_name)
                if device.carrier not in carrier_encoder.classes_:
                    carrier_encoder.classes_ = np.append(carrier_encoder.classes_, device.carrier)

                encoded_device = device_encoder.transform([device.device_name])[0]
                encoded_carrier = carrier_encoder.transform([device.carrier])[0]
                input_data = [encoded_device, encoded_carrier]
            else:
                raise ValueError("Device name or carrier is empty")
        except Exception as ve:
            print(f"Input data error: {str(ve)}")
            raise HTTPException(status_code=400, detail=f"Invalid input data: {str(ve)}")

        # AI 모델로 예측 수행
        try:
            if hasattr(model, 'predict'):
                predicted_result = model.predict([input_data])[0]
            elif isinstance(model, np.ndarray):
                predicted_result = np.dot(model, input_data)
            else:
                raise ValueError("Unsupported model type")

            # 모델 출력 처리
            if isinstance(predicted_result, (int, float)):
                predicted_score = float(predicted_result)
            elif isinstance(predicted_result, str):
                # 문자열 출력을 숫자로 매핑
                score_mapping = {
                    "ProductISPISP": 20,  # 예시 값, 실제 상황에 맞게 조정 필요
                    "OtherCategory": 15,
                    # 다른 가능한 출력들에 대한 매핑 추가
                }
                predicted_score = score_mapping.get(predicted_result, 0)  # 기본값 0
            else:
                raise ValueError(f"Unexpected model output type: {type(predicted_result)}")

        except Exception as model_error:
            print(f"Model prediction error: {str(model_error)}")
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(model_error)}")



        # 위험도 판단
        risk_level = None
        for level, data in risk_levels.items():
            if data["min_score"] <= predicted_score <= data["max_score"]:
                risk_level = data
                break

        if risk_level is None:
            raise ValueError(f"Invalid predicted score: {predicted_score}")

        # Supabase에 예측 결과 저장
        prediction_data = {
            "device_name": device.device_name,
            "carrier": device.carrier,
            "predicted_score": predicted_score,
            "risk_level": risk_level["name"],
            "risk_color": risk_level["color"],
            "risk_description": risk_level["description"],
            "ip_address": client_ip
        }

        supabase.table("predictions").insert(prediction_data).execute()

        return prediction_data


    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in create_device: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/predictions/latest")
async def get_latest_prediction():
    try:
        response = supabase.table("predictions").select("*").order('created_at', desc=True).limit(1).execute()
        if response.data:
            return response.data[0]
        else:
            raise HTTPException(status_code=404, detail="No predictions found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 라우터를 앱에 포함
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "환영합니다."}

# 서버 실행
# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)