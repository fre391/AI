from crewai_tools import BaseTool
import requests
import json

class OsmQueryTool(BaseTool):
    name: str = "OpenStreetMap Query Tool"
    description: str = "Query OpenStreetMap by string. Use this tool to get real-time data from OpenStreetMap. It delivers a JSON string with the results of the query."
    api_url: str = "https://www.overpass-api.de/api/interpreter"
    timeout: int = 10  # seconds
    cache: dict = {}

    def _parse_response(self, response: bytes) -> list:
        return response

    def _run(self, query: str) -> list:
        query_string = query  # Use the provided query
        query_key = f"overpass:{query_string}"
        
        if query_key in self.cache:
            return self.cache[query_key]

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {"data": query_string}
        print(f"Sending request with data: {data}")
        
        try:
            response = requests.post(self.api_url, headers=headers, data=data, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return []

        result = response.content
        self.cache[query_key] = result  # Cache the result
        return self._parse_response(result)

    def clear_cache(self) -> None:
        # Clear the cache
        self.cache = {}
