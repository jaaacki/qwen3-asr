import unittest
from fastapi.testclient import TestClient

# Mock out _ensure_worker so it doesn't spin up a subprocess
import sys
from unittest.mock import AsyncMock, patch

patcher = patch('gateway._ensure_worker', new_callable=AsyncMock)
patcher.start()

from gateway import app

class TestGatewaySwaggerDocs(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_swagger_ui_reachable(self):
        """Test that the Swagger UI HTML page correctly serves at /docs"""
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Swagger UI", response.text)

    def test_openapi_schema_contains_translation_endpoint(self):
        """Test that the OpenAPI JSON schema includes /v1/audio/translations"""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        
        schema = response.json()
        paths = schema.get("paths", {})
        
        # Verify the endpoint exists
        self.assertIn("/v1/audio/translations", paths)
        
        # Verify method and summary
        translation_endpoint = paths["/v1/audio/translations"]["post"]
        self.assertEqual(translation_endpoint["summary"], "Translate Audio")
        
        # Verify parameters are defined (file, language, response_format)
        content_type = translation_endpoint["requestBody"]["content"]["multipart/form-data"]
        schema_ref = content_type["schema"]["$ref"]
        schema_name = schema_ref.split("/")[-1]
        
        # Pull properties from the component schema
        properties = schema.get("components", {}).get("schemas", {})[schema_name]["properties"]
        
        self.assertIn("file", properties)
        self.assertIn("language", properties)
        self.assertIn("response_format", properties)

    def test_openapi_schema_contains_pydantic_models(self):
        """Test that the OpenAPI JSON schema registers the new Pydantic models from schemas.py"""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        
        schema = response.json()
        components = schema.get("components", {}).get("schemas", {})
        
        # Gateway uses TranscriptionResponse explicit model
        self.assertIn("TranscriptionResponse", components)

if __name__ == '__main__':
    unittest.main()
