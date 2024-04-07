package com.a303.notificationms.domain.memberToken;

import com.a303.notificationms.domain.MongoBaseEntity;
import java.util.List;
import java.util.Set;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.DBRef;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

@Getter
@Setter
@Document(collection = "member_token")
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class MemberToken extends MongoBaseEntity {

	@Id
	private String id;

	@Field(name = "member_id")
	private int memberId;

	@Field(name = "tokens")
	private Set<String> tokens;

	public static MemberToken createMemberToken(int memberId, Set<String> tokens) {
		MemberToken memberToken = new MemberToken();

		memberToken.setMemberId(memberId);
		memberToken.setTokens(tokens);

		return memberToken;
	}

}
