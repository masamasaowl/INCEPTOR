#!/usr/bin/env python3
"""
Command Line Interface for Voice Authentication System
Provides easy testing and management of the voice authentication system
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import settings, ensure_directories
from core.service import voice_auth_service
from database.manager import db_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoiceAuthCLI:
    """Command-line interface for voice authentication operations"""
    
    def __init__(self):
        """Initialize CLI"""
        self.service = voice_auth_service
        self.db = db_manager
    
    async def setup(self):
        """Setup database and directories"""
        ensure_directories()
        await self.db.initialize_database()
        logger.info("System initialized")
    
    async def register_user(self, username: str, email: Optional[str] = None):
        """Register a new user"""
        print(f"\n📝 Registering user: {username}")
        result = await self.service.register_user(username, email)
        
        if result['success']:
            print(f"✅ {result['message']}")
            print(f"   User UUID: {result['user_uuid']}")
            print(f"   Samples required: {result['samples_required']}")
        else:
            print(f"❌ Error: {result['error']}")
    
    async def add_voice_sample(self, username: str, audio_path: str, passphrase: str):
        """Add a voice sample for enrollment"""
        print(f"\n🎤 Adding voice sample for: {username}")
        
        audio_file = Path(audio_path)
        if not audio_file.exists():
            print(f"❌ Error: Audio file not found: {audio_path}")
            return
        
        result = await self.service.add_voice_sample(username, audio_path, passphrase)
        
        if result['success']:
            print(f"✅ {result['message']}")
            print(f"   Samples: {result['samples_collected']}/{result['samples_required']}")
            if 'quality_metrics' in result:
                metrics = result['quality_metrics']
                print(f"   Duration: {metrics['duration']:.2f}s")
                print(f"   SNR: {metrics['snr_db']:.1f} dB")
            if result['fully_registered']:
                print(f"   🎉 User fully registered and ready to authenticate!")
        else:
            print(f"❌ Error: {result['error']}")
    
    async def authenticate(self, username: str, audio_path: str, passphrase: str):
        """Authenticate a user"""
        print(f"\n🔐 Authenticating user: {username}")
        
        audio_file = Path(audio_path)
        if not audio_file.exists():
            print(f"❌ Error: Audio file not found: {audio_path}")
            return
        
        result = await self.service.authenticate(username, audio_path, passphrase)
        
        if result['success']:
            if result['authenticated']:
                print(f"✅ AUTHENTICATION SUCCESSFUL")
                print(f"   Username: {result['username']}")
                print(f"   User UUID: {result['user_uuid']}")
                print(f"   Similarity Score: {result['similarity_score']:.3f}")
                print(f"   Threshold: {result['threshold']:.3f}")
            else:
                print(f"❌ AUTHENTICATION FAILED")
                print(f"   Similarity Score: {result['similarity_score']:.3f}")
                print(f"   Threshold: {result['threshold']:.3f}")
                print(f"   Message: {result['message']}")
        else:
            print(f"❌ Error: {result['error']}")
    
    async def get_user_info(self, username: str):
        """Get user information"""
        print(f"\n👤 User Information: {username}")
        result = await self.service.get_user_info(username)
        
        if result['success']:
            print(f"   Username: {result['username']}")
            print(f"   UUID: {result['user_uuid']}")
            print(f"   Email: {result['email'] or 'N/A'}")
            print(f"   Registered: {'Yes' if result['is_registered'] else 'No'}")
            print(f"   Samples: {result['samples_collected']}/{result['samples_required']}")
            print(f"   Created: {result['created_at']}")
            print(f"   Active: {'Yes' if result['is_active'] else 'No'}")
        else:
            print(f"❌ Error: {result['error']}")
    
    async def list_users(self):
        """List all users"""
        print(f"\n👥 All Users:")
        result = await self.service.list_users()
        
        if result['success']:
            if result['total_users'] == 0:
                print("   No users found")
            else:
                print(f"   Total: {result['total_users']}")
                print()
                for user in result['users']:
                    status = "✓ Registered" if user['is_registered'] else "⚠ Incomplete"
                    print(f"   • {user['username']}")
                    print(f"     UUID: {user['user_uuid']}")
                    print(f"     Status: {status} ({user['samples_collected']} samples)")
                    print(f"     Created: {user['created_at']}")
                    print()
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    async def delete_user(self, username: str):
        """Delete a user"""
        print(f"\n🗑️  Deleting user: {username}")
        
        # Confirm deletion
        confirm = input(f"Are you sure you want to delete '{username}' and all their data? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Deletion cancelled")
            return
        
        result = await self.service.delete_user(username)
        
        if result['success']:
            print(f"✅ {result['message']}")
        else:
            print(f"❌ Error: {result['error']}")
    
    async def get_auth_history(self, username: str, limit: int = 10):
        """Get authentication history"""
        print(f"\n📊 Authentication History: {username}")
        result = await self.service.get_authentication_history(username, limit)
        
        if result['success']:
            if result['total_attempts'] == 0:
                print("   No authentication attempts found")
            else:
                print(f"   Total attempts: {result['total_attempts']}")
                print()
                for i, attempt in enumerate(result['history'], 1):
                    status = "✅ SUCCESS" if attempt['success'] else "❌ FAILED"
                    print(f"   {i}. {attempt['timestamp']} - {status}")
                    print(f"      Score: {attempt['similarity_score']:.3f}")
                    print(f"      Passphrase: {attempt['passphrase']}")
                    if attempt['failure_reason']:
                        print(f"      Reason: {attempt['failure_reason']}")
                    print()
        else:
            print(f"❌ Error: {result['error']}")
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.db.close()


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Voice Authentication System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register a new user
  python cli.py register --username john_doe --email john@example.com
  
  # Add voice samples (repeat 3 times with same passphrase)
  python cli.py enroll --username john_doe --audio voice1.wav --passphrase "Hello world"
  python cli.py enroll --username john_doe --audio voice2.wav --passphrase "Hello world"
  python cli.py enroll --username john_doe --audio voice3.wav --passphrase "Hello world"
  
  # Authenticate
  python cli.py auth --username john_doe --audio test.wav --passphrase "Hello world"
  
  # View user info
  python cli.py info --username john_doe
  
  # List all users
  python cli.py list
  
  # View authentication history
  python cli.py history --username john_doe
  
  # Delete user
  python cli.py delete --username john_doe
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a new user')
    register_parser.add_argument('--username', required=True, help='Username')
    register_parser.add_argument('--email', help='Email address (optional)')
    
    # Enroll command
    enroll_parser = subparsers.add_parser('enroll', help='Add voice sample')
    enroll_parser.add_argument('--username', required=True, help='Username')
    enroll_parser.add_argument('--audio', required=True, help='Path to audio file')
    enroll_parser.add_argument('--passphrase', required=True, help='What the user said')
    
    # Authenticate command
    auth_parser = subparsers.add_parser('auth', help='Authenticate user')
    auth_parser.add_argument('--username', required=True, help='Username')
    auth_parser.add_argument('--audio', required=True, help='Path to audio file')
    auth_parser.add_argument('--passphrase', required=True, help='What the user said')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get user information')
    info_parser.add_argument('--username', required=True, help='Username')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all users')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete user')
    delete_parser.add_argument('--username', required=True, help='Username')
    
    # History command
    history_parser = subparsers.add_parser('history', help='Get authentication history')
    history_parser.add_argument('--username', required=True, help='Username')
    history_parser.add_argument('--limit', type=int, default=10, help='Number of records')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create CLI instance
    cli = VoiceAuthCLI()
    
    try:
        # Setup
        await cli.setup()
        
        # Execute command
        if args.command == 'register':
            await cli.register_user(args.username, args.email)
        
        elif args.command == 'enroll':
            await cli.add_voice_sample(args.username, args.audio, args.passphrase)
        
        elif args.command == 'auth':
            await cli.authenticate(args.username, args.audio, args.passphrase)
        
        elif args.command == 'info':
            await cli.get_user_info(args.username)
        
        elif args.command == 'list':
            await cli.list_users()
        
        elif args.command == 'delete':
            await cli.delete_user(args.username)
        
        elif args.command == 'history':
            await cli.get_auth_history(args.username, args.limit)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
    finally:
        await cli.cleanup()


if __name__ == "__main__":
    asyncio.run(main())