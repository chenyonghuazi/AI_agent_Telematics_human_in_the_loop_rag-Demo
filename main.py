from langgraph.types import interrupt, Command
from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.prelude import Send
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model

from langgraph.graph.state import CompiledStateGraph


from ragPipeline.rag_pipeline import rag_Pipeline
import operator
from dotenv import load_dotenv
import pandas as pd
from geopy.distance import geodesic
from src.model import Query
import json
import os

load_dotenv()

rag = rag_Pipeline()

# =====================================
# 1. DriverReportState 
# =====================================
class DriverReportState(TypedDict):
    # user input state
    driver_id: str
    start_date: str                # e.g. "2026-03-09"
    end_date: str                  # e.g. "2026-03-15"
    raw_data: pd.DataFrame           
    
    # middle state
    metrics_summary: dict          
    analysis_result: dict          
    anomalies: list[dict]          
    anomaly_level: str            
    report_draft: str              
    messages: Annotated[Sequence[BaseMessage], operator.add]  # history message

    # final state
    final_report: Literal["Pass", "Fail"] | None
    human_feedback: str | None
    
# =====================================
# 2. Tools
# =====================================
def read_file() -> pd.DataFrame:
    """read file"""
    df = pd.read_csv("src/Telematicsdata.csv")
    
    df['timeMili'] = pd.to_datetime(df['timeMili'], unit='ms')
    
    # print(df)
    
    return df
  
# read_file()

# calculate distance by gps coordinates    
def calculate_distance_by_row(row):
    # last gps coordinate
    lat1, lon1 = row['prev_lat'], row['prev_lon']
    lat2, lon2 = row['latitude'], row['longitude']
    
    # if first line, then distance = 0.0
    if pd.isna(lat1):
        return 0.0
    
    
    return geodesic((lat1, lon1), (lat2, lon2)).meters

# @tool
def query_telematics_data(driver_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """grabs telematics data for a given driver and date range"""
    
    df = read_file()
    
   
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # 包含 end 当天最后一秒
    
    df = df[(df['deviceId'] == driver_id) & (df['timeMili'] >= start) & (df['timeMili'] <= end)]

    return df



def calculate_distance(df: pd.DataFrame) -> float:
    """calculate distance"""
    
    df = df[df['variable'] == 'POSITION']
    
    df['latitude'] = df['value'].str.split(',').str[0].astype(float)
    df['longitude'] = df['value'].str.split(',').str[1].astype(float)
    
    df['prev_lat'] = df['latitude'].shift(1)
    df['prev_lon'] = df['longitude'].shift(1)
    
    # speed could be slow if the dataset is large, consider using a more efficient method or parallel processing
    df['distance_m'] = df.apply(calculate_distance_by_row, axis=1)
    
    total_distance = df['distance_m'].sum()
    
    print(df[['timeMili', 'latitude', 'longitude', 'distance_m']])
    
    
    return total_distance

print(calculate_distance(query_telematics_data('zRYzhAEAHAABAAAKCRtcAAsAtwB1gBAQ','2020-08-16','2020-08-17')))

def calculate_average_speed(df: pd.DataFrame) -> float:
    """计算平均速度"""
    df = df[df['variable'] == 'Vehicle speed']
    df['speed'] = df['value'].astype(float)
    average_speed = df['speed'].mean()
    return average_speed

print(calculate_average_speed(query_telematics_data('zRYzhAEAHAABAAAKCRtcAAsAtwB1gBAQ','2020-08-16','2020-08-17')))

def count_over_speed(df: pd.DataFrame) -> int:
    """计算超速次数"""
    df = df[df['variable'] == 'Vehicle speed']
    df['speed'] = df['value'].astype(float)
    over_speed_trips = df[df['speed'] > 80].shape[0]
    return over_speed_trips

print(count_over_speed(query_telematics_data('zRYzhAEAHAABAAAKCRtcAAsAtwB1gBAQ','2020-08-16','2020-08-17')))

def count_idling_time(df: pd.DataFrame) -> int:
    """计算怠速时长"""
    df = df[(df['variable'] == 'IDLING') & (df['value'] == 1)]
    
    idling_time = df.shape[0]
    return idling_time
print(count_idling_time(query_telematics_data('zRYzhAEAHAABAAAKCRtcAAsAtwB1gBAQ','2020-08-16','2020-08-17')))

# optional
# @tool 
# def send_email_notification(driver_id: str, report_summary: str, severity: str):
#     """send email notification"""
#     
#     return "Notification sent"

# ==============================tools  end================================================

# ===============================node moethod================================================

def fetechAndSummary(state: DriverReportState) -> DriverReportState:
    """summarize dataset"""
    # 1. original data
    df = query_telematics_data(state['driver_id'], state['start_date'], state['end_date'])
    
    
    
    # 2. summary
    state['metrics_summary'] = {
        "total_distance": float(calculate_distance(df)),  # m
        "average_speed": float(calculate_average_speed(df)),     # km/h
        "overspeed_events": count_over_speed(df),
        "idle_time": count_idling_time(df)
    }
    
    # print(state['metrics_summary'])
    
    return state

def detectAnomalies(state: DriverReportState) -> DriverReportState:
    
    context = rag.rerank("rule for vehicle speed and distance",rag.retrieve("rule for vehicle speed and distance",5),3)
     
    
    prompt = ChatPromptTemplate.from_messages([
        ('system',"""
         You are a professional driver safety expert. You are given a list of events and a context. Your task is to detect any anomaly in the events.
         If there is an anomaly, you should return a list of anomalies.
         
         Must be follwing these rules otherwise task will fail: 
                1. Output only pure JSON objects; no prefixes, suffixes, markdown, or ```json` are allowed.
                2. All double quotes in the string must be escaped as `\"` (backslash followed by a double quote). Single quotes do not need to be escaped. Backslashes `**` must be escaped as `\\**`.
                3. Content in the `event_value` field can have line breaks, but the line break character must be `\n`.
                4. Unescaped double quotes, line breaks, control characters, and other content that could disrupt the JSON structure are prohibited.
                5. JSON strings must be completely closed; commas and parentheses are not allowed.
                
        Output:
        - "evaluation": 'Good' | 'Bad' | 'moderate'
        - "event": for each event, we need to provide the event type and event value.
         
         """
         ),
        ('human',""" Driver data: {input_data}, analyze any anomaly based on context: {input_context}.
         """)
    ])
    
    llm = init_chat_model("Qwen/Qwen3-8B",
                        model_provider="openai",
                        base_url="https://api.siliconflow.cn/v1",
                        api_key=os.getenv("SILICONFLOW_API_KEY"),
                        temperature=0.5,
                        max_tokens=8192
                        )
    
    chain = (
        RunnablePassthrough.assign(
            input_data = lambda _: json.dumps(state['metrics_summary'], ensure_ascii=False, indent=2),
            input_context = lambda x: x["input_context"]
        )
        | prompt
        | llm.with_structured_output(Query)
    )
    
    result = chain.invoke({"input_context": context})
    # print(result)
    # print(type(result))
    # print(json.loads(result.model_dump_json()))
    return {"analysis_result":json.loads(result.model_dump_json())}

def human_review(state: DriverReportState) -> DriverReportState:

    
    content = state['analysis_result']
    
    if content.get('evaluation') == 'bad':
        
        raise interrupt(value ={
            "question": "Please review this driver report",
            "content_to_review": f"Please review this driver report: {state['analysis_result']}",
            "options": ["approve", "reject", "request_changes"],
            "require_reason": True   # optional
        })
             
    # it will not run this return if we need human review   
    return {'final_report': 'Pass'}

def after_human_review(state:DriverReportState) -> DriverReportState:
    print(f"Human review completed. Final report: {state['final_report']}, Feedback: {state.get("human_feedback","")}")
    
    return state

# ================================method end=================================================================
# ================================Langgraph workflow==========================================================

def run_with_human_in_the_loop(graph: CompiledStateGraph, config: dict, initial_input: dict):
    inputs = initial_input.copy()
    first_time = True
    while True:
        stream_input = inputs if first_time else None
        # print updates
        for chunk in graph.stream(stream_input, config, stream_mode="updates"):
            print("→", chunk)
        
        # when it stop, get the state
        current_state = graph.get_state(config)
        # print(f"current state: {current_state}")
        
                # 3rd siuation, graph is interrupted
        if current_state.interrupts:
            interrupt_data = current_state.interrupts[0].value   # we only have one interrupt in the workflow
            try:
                if not isinstance(interrupt_data, dict):
                    interrupt_data = {"question": str(interrupt_data)}
            except Exception as e:
                print(e)
                raise ValueError("Interrupt data must be a dict")
            
            print("\n" + "="*50)
            print("Need human's review：")
            print("Question：", interrupt_data.get("question", "question requestion failed"))
            if "content_to_review" in interrupt_data:
                print("content to review：")
                print(interrupt_data["content_to_review"])
            print("Options：", interrupt_data.get("options", ["approve", "reject", "request_changes"]))
            print("="*50 + "\n")
            
            # ------------------ Human Input------------------
            # couldbe front-end、API、Slack、email reply、librart flag .....etc
            human_input = input("Please input your decision（approve, reject, request_changes）： ")
            # human_input = await get_feedback_from_api()   # real case
            
            # Command is recommended to use in new langgraph
            # resume_command = Command(
            #     resume={
            #         "human_feedback": human_input,
            #         "final_report": "Pass" if "approve" in human_input else 'Fail'
            #     }
            # )
            
            # update state
            graph.update_state(
                config,
                values={
                    "human_feedback": human_input,
                    "final_report": "Pass" if "approve" in human_input else 'Fail'
                }
            )
            first_time = False
            print("human input done and continue...\n")
            continue
        
        # 1st siuation, graph is complete and nothing wrong
        if not current_state.next and not current_state.interrupts:
            print("Workflow is done")
            return current_state.values
        
        # 2nd siuation, graph is running 
        if current_state.next:
            print("Next node...")
            continue
        

        
        

if __name__ == "__main__":

    app = StateGraph(DriverReportState)

    app.add_node("fetch_data",fetechAndSummary)
    app.add_node("detect_anomalies",detectAnomalies)
    app.add_node("human_review",human_review)
    app.add_node("after_human_review",after_human_review)

    app.add_edge(START, "fetch_data")
    app.add_edge("fetch_data", "detect_anomalies")
    app.add_edge("detect_anomalies", "human_review")
    app.add_edge("human_review", "after_human_review")
    app.add_edge("after_human_review", END)

    memory = MemorySaver()
    agent = app.compile(checkpointer=memory)




    initial_state = {'driver_id':'zRYzhAEAHAABAAAKCRtcAAsAtwB1gBAQ','start_date':'2020-08-16','end_date':'2020-08-17'}

    thread = {"configurable": {"thread_id": "daily_report_zRYzhAEAHAABAAAKCRtcAAsAtwB1gBAQ"}}

    result = run_with_human_in_the_loop(agent, thread, initial_state)

    # print(result)
  


