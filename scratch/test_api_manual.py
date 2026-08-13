import urllib.request
import docx
import os

def test_api():
    boundary = '----Boundary'
    headers = {
        'Content-Type': f'multipart/form-data; boundary={boundary}'
    }
    
    with open('tests/split_run_sample.docx', 'rb') as f:
        data = f.read()
        
    body = b'\r\n'.join([
        f'--{boundary}'.encode(),
        b'Content-Disposition: form-data; name="file"; filename="split_run_sample.docx"',
        b'Content-Type: application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        b'',
        data,
        f'--{boundary}--'.encode()
    ])
    
    print("Uploading to http://127.0.0.1:8000/redact...")
    req = urllib.request.Request('http://127.0.0.1:8000/redact', data=body, headers=headers)
    try:
        with urllib.request.urlopen(req) as res:
            out_bytes = res.read()
            print(f"Success! Response size: {len(out_bytes)} bytes.")
            
            output_path = 'outputs/api_redacted.docx'
            with open(output_path, 'wb') as out_f:
                out_f.write(out_bytes)
            
            doc = docx.Document(output_path)
            print(f"Redacted document verified! Opened successfully. Paragraphs: {len(doc.paragraphs)}")
            
            # Clean up the output test file
            if os.path.exists(output_path):
                os.unlink(output_path)
                print("Cleaned up temporary redacted file.")
    except Exception as e:
        print("API test failed:", e)

if __name__ == '__main__':
    test_api()
