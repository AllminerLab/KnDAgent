#!/usr/bin/env python3
"""
Medical Knowledge Graph Recommendation Agent startup script
"""

import os
import sys
from medical_agent_v2 import MedicalKnowledgeAgent

def main():
    """Main startup function"""
    print("🏥 Medical Knowledge Graph Recommendation Agent V2")
    print("=" * 60)
    
    # Check necessary files
    doc_path = "./data/KnDAgent.docx"
    
    if not os.path.exists(doc_path):
        print(f"❌ Error: Document path does not exist: {doc_path}")
        print("Please ensure medical knowledge document is placed in the correct location")
        return
    
    print("✅ File check completed, starting Agent...")
    print()
    
    try:
        # Create and start agent
        agent = MedicalKnowledgeAgent(doc_path=doc_path)
        
        # Start interactive mode
        agent.interactive_mode()
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted, exiting...")
    except Exception as e:
        print(f"\n❌ Agent startup failed: {e}")
        print("Please check error information and retry")

if __name__ == "__main__":
    main()
