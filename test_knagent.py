#!/usr/bin/env python3
"""
KnDAgent test script
"""

import os
import sys
import time
from knagent import KnDAgent

def test_basic_functionality():
    """Test basic functionality"""
    print("🧪 Starting basic functionality test...")
    
    # Test cases for emergency tasks
    test_cases = [
        "Emergency triage for patient with chest pain and shortness of breath",
        "DDI detection for patient taking warfarin and aspirin",
        "Readmission prediction for elderly patient with heart failure",
        "Medication recommendation for patient with hypertension and diabetes",
        "Emergency triage for patient with severe abdominal pain"
    ]
    
    try:
        # Create agent
        print("Initializing KnDAgent...")
        agent = KnDAgent()
        
        # Test each case
        for i, task in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"Test Case {i}: {task}")
            print(f"{'='*60}")
            
            start_time = time.time()
            execution_chain = agent.process_emergency_task(task)
            end_time = time.time()
            
            print(f"Execution Chain (Time: {end_time - start_time:.2f}s):")
            print(f"Task ID: {execution_chain.task_info.task_id}")
            print(f"Task Type: {execution_chain.task_info.task_type}")
            print(f"Priority: {execution_chain.task_info.priority}")
            print(f"Selected Graph: {execution_chain.selected_graph}")
            print(f"Selected Tools: {', '.join(execution_chain.selected_tools)}")
            print(f"Confidence Score: {execution_chain.confidence_score:.2f}")
            
            # Simple validation
            if execution_chain.selected_graph and execution_chain.selected_tools:
                print("✅ Execution chain format correct")
            else:
                print("❌ Execution chain format abnormal")
            
            time.sleep(1)  # Avoid too frequent requests
        
        print("\n🎉 Basic functionality test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_rag_system():
    """Test RAG system"""
    print("\n🔍 Starting RAG system test...")
    
    try:
        agent = KnDAgent()
        
        # Test document processing
        if agent.doc_processor.sections:
            print(f"✅ Document processing successful, total {len(agent.doc_processor.sections)} sections")
        else:
            print("❌ Document processing failed")
            return False
        
        # Test RAG retrieval
        test_query = "emergency triage"
        relevant_context = agent.rag_system.retrieve_relevant_context(test_query)
        
        if relevant_context:
            print(f"✅ RAG retrieval successful, found {len(relevant_context)} relevant chunks")
            for i, context in enumerate(relevant_context[:2], 1):
                print(f"  Chunk {i}: {context[:100]}...")
        else:
            print("❌ RAG retrieval failed")
            return False
        
        print("🎉 RAG system test completed!")
        return True
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False

def test_knowledge_graphs():
    """Test knowledge graph library"""
    print("\n📚 Starting knowledge graph library test...")
    
    try:
        agent = KnDAgent()
        
        if agent.knowledge_graphs:
            print(f"✅ Knowledge graph library initialization successful, total {len(agent.knowledge_graphs)} graphs")
            
            for i, kg in enumerate(agent.knowledge_graphs, 1):
                print(f"  {i}. {kg.name} - {kg.domain}")
                print(f"     Use Cases: {', '.join(kg.use_cases[:2])}")
        else:
            print("❌ Knowledge graph library initialization failed")
            return False
        
        print("🎉 Knowledge graph library test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Knowledge graph library test failed: {e}")
        return False

def test_model_integration():
    """Test model integration"""
    print("\n🤖 Starting model integration test...")
    
    try:
        agent = KnDAgent()
        
        # Test simple generation
        test_prompt = "Please briefly explain the role of medical knowledge graphs"
        response = agent.llm.generate_text(
            prompt=test_prompt,
            max_tokens=100,
            temperature=0.7
        )
        
        if response and len(response) > 10:
            print("✅ Model generation successful")
            print(f"   Response: {response[:100]}...")
        else:
            print("❌ Model generation failed")
            return False
        
        print("🎉 Model integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🏥 KnDAgent Test Suite")
    print("=" * 60)
    
    # Check necessary files
    required_files = [
        "./data/KnDAgent.docx"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing required file: {file_path}")
            print("Please ensure all required files are in place")
            return
    
    print("✅ File check completed")
    
    # Run tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("RAG System", test_rag_system),
        ("Knowledge Graph Library", test_knowledge_graphs),
        ("Model Integration", test_model_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test exception: {e}")
    
    # Test results summary
    print("\n" + "="*60)
    print("📊 Test Results Summary")
    print("="*60)
    print(f"Total Tests: {total}")
    print(f"Passed Tests: {passed}")
    print(f"Failed Tests: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! System running normally")
    else:
        print(f"\n⚠️  {total-passed} tests failed, please check system configuration")
    
    print("="*60)

if __name__ == "__main__":
    main()
